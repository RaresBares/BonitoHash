#!/bin/bash
###############################################################################
# BonitoHash — SAFARI Cluster Befehlsliste (Command Guide)
#
# Ablauf:  Clone → Setup → Daten → Train Vanilla → Train CrossAttn
#          → Basecall+Eval D1 (SARS-CoV-2) → Vergleich
#          → wenn besser: D2 (E. coli) → Vergleich
#          → wenn besser: D5 (Human HG001) → Vergleich
#
# WICHTIG: Ersetze NETHZ mit deinem ETH-Benutzernamen (z.B. rbares)
###############################################################################

# ============================================================================
# 0) SSH zum Cluster
# ============================================================================
# Von deinem lokalen Terminal:
ssh safari-proxy                  # oder: ssh NETHZ@safari-proxy.ethz.ch

# ============================================================================
# 1) Arbeitsverzeichnis anlegen (NICHT unter /home, sondern /mnt/galactica!)
# ============================================================================
export WORK=/mnt/galactica/$USER/BonitoHash
mkdir -p $WORK
cd $WORK

# ============================================================================
# 2) Repo klonen
# ============================================================================
git clone https://github.com/RaresBares/BonitoHash.git
cd BonitoHash

# ============================================================================
# 3) Python venv einrichten (kein conda auf dem Cluster)
# ============================================================================
# Prüfe welche Python-Version vorhanden ist:
python3 --version
# Falls python3.12 nicht default ist, suche es:
# ls /usr/bin/python3*
# oder: which python3.12

# venv erstellen
python3 -m venv $WORK/BonitoHash/venv

# venv aktivieren
source $WORK/venv/bin/activate

# pip aktualisieren
pip install --upgrade pip

# CUDA-Version prüfen (auf GPU-Nodes)
# ls /usr/local/cuda*
# export PATH="/usr/local/cuda-12.2/bin:$PATH"  # falls nötig

# PyTorch mit CUDA installieren
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Bonito + Abhängigkeiten installieren
pip install -e .

# ont-koi (spezielle Bonito-Abhängigkeit — braucht evtl. CUDA auf GPU-Node)
# Falls koi hier fehlschlägt, auf GPU-Node installieren (siehe Schritt 3b)

# ============================================================================
# 3b) Installation auf GPU-Node testen (interaktiv)
# ============================================================================
srun -p gpu_part --gres gpu:1 -c 8 --mem=32GB -D $WORK --pty bash -l
source $WORK/venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import bonito; print('bonito OK')"
# Ctrl+D zum Verlassen

# ============================================================================
# 4) Daten herunterladen
# ============================================================================
export DATA=$WORK/data
mkdir -p $DATA

# --- 4a) Bonito Trainingsdaten (für Modell-Training) ---
source $WORK/venv/bin/activate
bonito download --training --out_dir $DATA/training

# --- 4b) Referenzgenome herunterladen ---
mkdir -p $DATA/references

# SARS-CoV-2 (D1)
wget -O $DATA/references/sars_cov2.fa.gz \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/858/895/GCF_009858895.2_ASM985889v3/GCF_009858895.2_ASM985889v3_genomic.fna.gz"
gunzip $DATA/references/sars_cov2.fa.gz

# E. coli K12 (D2)
wget -O $DATA/references/ecoli_k12.fa.gz \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz"
gunzip $DATA/references/ecoli_k12.fa.gz

# Human GRCh38 (D5)
wget -O $DATA/references/grch38.fa.gz \
  "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
gunzip $DATA/references/grch38.fa.gz

# --- 4c) ONT k-mer Pore Model (für Cross-Attention) ---
mkdir -p $DATA/kmer_models
# Das 6-mer Pore Model für r10.4.1 DNA von ONT's Rerio Repository:
wget -O $DATA/kmer_models/r10.4.1_6mer.tsv \
  "https://raw.githubusercontent.com/nanoporetech/kmer_models/master/r10.4.1_400bps.nucleotide.6mer.template.model"
# Falls URL nicht mehr aktuell — Alternative:
# git clone https://github.com/nanoporetech/kmer_models.git $DATA/kmer_models/repo
# cp $DATA/kmer_models/repo/r10.4.1_400bps.nucleotide.6mer.template.model $DATA/kmer_models/r10.4.1_6mer.tsv

# --- 4d) Nanopore Reads (POD5/FAST5) ---
# SARS-CoV-2 ONT Reads (ARTIC-Protokoll, z.B. von ENA/SRA)
mkdir -p $DATA/reads/sars_cov2
# Beispiel — passe SRA-Accession an dein gewünschtes Dataset an:
# Variante 1: Direkt von ONT Open Data
# Variante 2: SRA
# prefetch SRR... && fasterq-dump SRR... -O $DATA/reads/sars_cov2/
# Variante 3: Falls du schon POD5-Dateien hast, kopiere sie hierhin

# E. coli ONT Reads
mkdir -p $DATA/reads/ecoli
# Beispiel: ONT Open Data E. coli dataset
# wget ... -O $DATA/reads/ecoli/

# Human HG001 (NA12878) ONT Reads
mkdir -p $DATA/reads/human
# ONT Open Data:
# https://labs.epi2me.io/open-data/ — suche nach NA12878 / HG001

echo "==> Bitte fülle die Reads-Verzeichnisse mit deinen POD5/FAST5 Dateien!"
echo "==> Die obigen wget-Befehle sind Platzhalter — passe sie an deine Datenquelle an."

# ============================================================================
# 5) RawHash Mappings erstellen (für Cross-Attention Basecalling)
# ============================================================================
# RawHash muss separat installiert werden
mkdir -p $WORK/tools
cd $WORK/tools
git clone https://github.com/CMU-SAFARI/RawHash.git
cd RawHash && make
export PATH=$WORK/tools/RawHash/bin:$PATH
cd $WORK/BonitoHash

# RawHash für jedes Dataset laufen lassen:
mkdir -p $DATA/rawhash

# SARS-CoV-2
rawhash -t 8 $DATA/references/sars_cov2.fa $DATA/reads/sars_cov2/ \
  > $DATA/rawhash/sars_cov2.paf 2> $DATA/rawhash/sars_cov2.log

# E. coli
rawhash -t 8 $DATA/references/ecoli_k12.fa $DATA/reads/ecoli/ \
  > $DATA/rawhash/ecoli.paf 2> $DATA/rawhash/ecoli.log

# Human (nur wenn D1/D2 besser sind)
rawhash -t 8 $DATA/references/grch38.fa $DATA/reads/human/ \
  > $DATA/rawhash/human.paf 2> $DATA/rawhash/human.log

# ============================================================================
# 6) Training: Vanilla Baseline
# ============================================================================
cd $WORK/BonitoHash

cat > $WORK/train_vanilla.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=bonito-vanilla
#SBATCH --partition=gpu_part
#SBATCH --gres=gpu:A100-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export WORK=/mnt/galactica/$USER/bonitohash
cd $WORK/BonitoHash
source $WORK/venv/bin/activate

DATA=$WORK/data

# Vanilla Training (Standard-Config, kein k-mer Model)
bonito train $WORK/runs/vanilla \
  --pretrained dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
  --directory $DATA/training/example_data_dna_r10.4.1_v0 \
  --epochs 5 \
  --batch 64 \
  --lr 2e-3 \
  --device cuda \
  -f
SLURM_EOF

sbatch -D $WORK $WORK/train_vanilla.sh
# Job-Status prüfen: squeue -u $USER

# ============================================================================
# 7) Training: Cross-Attention Model
# ============================================================================
cat > $WORK/train_crossattn.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=bonito-crossattn
#SBATCH --partition=gpu_part
#SBATCH --gres=gpu:A100-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export WORK=/mnt/galactica/$USER/bonitohash
cd $WORK/BonitoHash
source $WORK/venv/bin/activate

DATA=$WORK/data

# Cross-Attention Training (mit k-mer Model + Cross-Attention Config)
bonito train $WORK/runs/crossattn \
  --config bonito/models/configs/dna_r10.4.1@v5.0_crossattn.toml \
  --directory $DATA/training/example_data_dna_r10.4.1_v0 \
  --kmer-model $DATA/kmer_models/r10.4.1_6mer.tsv \
  --epochs 5 \
  --batch 64 \
  --lr 2e-3 \
  --device cuda \
  -f
SLURM_EOF

sbatch -D $WORK $WORK/train_crossattn.sh

# ============================================================================
# 8) Basecalling + Evaluation: D1 — SARS-CoV-2
# ============================================================================
cat > $WORK/eval_d1.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=eval-d1-sarscov2
#SBATCH --partition=gpu_part
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export WORK=/mnt/galactica/$USER/bonitohash
cd $WORK/BonitoHash
source $WORK/venv/bin/activate

DATA=$WORK/data
RESULTS=$WORK/results/d1_sars_cov2
mkdir -p $RESULTS

REF=$DATA/references/sars_cov2.fa
READS=$DATA/reads/sars_cov2

# --- Vanilla Basecalling ---
bonito basecaller $WORK/runs/vanilla $READS \
  --reference $REF \
  --device cuda \
  > $RESULTS/vanilla.bam 2> $RESULTS/vanilla.log

# --- Cross-Attention Basecalling (mit RawHash) ---
bonito basecaller $WORK/runs/crossattn $READS \
  --reference $REF \
  --rawhash-paf $DATA/rawhash/sars_cov2.paf \
  --kmer-model $DATA/kmer_models/r10.4.1_6mer.tsv \
  --ref-fasta $REF \
  --device cuda \
  > $RESULTS/crossattn.bam 2> $RESULTS/crossattn.log

# --- Accuracy-Statistiken mit samtools ---
echo "=== VANILLA ===" > $RESULTS/comparison.txt
samtools stats $RESULTS/vanilla.bam | grep ^SN >> $RESULTS/comparison.txt
echo "" >> $RESULTS/comparison.txt
echo "=== CROSS-ATTENTION ===" >> $RESULTS/comparison.txt
samtools stats $RESULTS/crossattn.bam | grep ^SN >> $RESULTS/comparison.txt

echo ""
echo "=== Ergebnis D1 (SARS-CoV-2) ==="
cat $RESULTS/comparison.txt
SLURM_EOF

# Erst starten, wenn Training fertig ist!
# Ersetze VANILLA_JOBID und CROSSATTN_JOBID mit den echten Job-IDs von squeue
sbatch -D $WORK --dependency=afterok:VANILLA_JOBID,CROSSATTN_JOBID $WORK/eval_d1.sh

# ============================================================================
# 9) Basecalling + Evaluation: D2 — E. coli (nur wenn D1 besser)
# ============================================================================
cat > $WORK/eval_d2.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=eval-d2-ecoli
#SBATCH --partition=gpu_part
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export WORK=/mnt/galactica/$USER/bonitohash
cd $WORK/BonitoHash
source $WORK/venv/bin/activate

DATA=$WORK/data
RESULTS=$WORK/results/d2_ecoli
mkdir -p $RESULTS

REF=$DATA/references/ecoli_k12.fa
READS=$DATA/reads/ecoli

# --- Vanilla ---
bonito basecaller $WORK/runs/vanilla $READS \
  --reference $REF \
  --device cuda \
  > $RESULTS/vanilla.bam 2> $RESULTS/vanilla.log

# --- Cross-Attention ---
bonito basecaller $WORK/runs/crossattn $READS \
  --reference $REF \
  --rawhash-paf $DATA/rawhash/ecoli.paf \
  --kmer-model $DATA/kmer_models/r10.4.1_6mer.tsv \
  --ref-fasta $REF \
  --device cuda \
  > $RESULTS/crossattn.bam 2> $RESULTS/crossattn.log

# --- Vergleich ---
echo "=== VANILLA ===" > $RESULTS/comparison.txt
samtools stats $RESULTS/vanilla.bam | grep ^SN >> $RESULTS/comparison.txt
echo "" >> $RESULTS/comparison.txt
echo "=== CROSS-ATTENTION ===" >> $RESULTS/comparison.txt
samtools stats $RESULTS/crossattn.bam | grep ^SN >> $RESULTS/comparison.txt

echo "=== Ergebnis D2 (E. coli) ==="
cat $RESULTS/comparison.txt
SLURM_EOF

sbatch -D $WORK $WORK/eval_d2.sh

# ============================================================================
# 10) Basecalling + Evaluation: D5 — Human HG001 (nur wenn D2 besser)
# ============================================================================
cat > $WORK/eval_d5.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=eval-d5-human
#SBATCH --partition=gpu_part
#SBATCH --gres=gpu:A100-80GB:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export WORK=/mnt/galactica/$USER/bonitohash
cd $WORK/BonitoHash
source $WORK/venv/bin/activate

DATA=$WORK/data
RESULTS=$WORK/results/d5_human
mkdir -p $RESULTS

REF=$DATA/references/grch38.fa
READS=$DATA/reads/human

# --- Vanilla ---
bonito basecaller $WORK/runs/vanilla $READS \
  --reference $REF \
  --device cuda \
  > $RESULTS/vanilla.bam 2> $RESULTS/vanilla.log

# --- Cross-Attention ---
bonito basecaller $WORK/runs/crossattn $READS \
  --reference $REF \
  --rawhash-paf $DATA/rawhash/human.paf \
  --kmer-model $DATA/kmer_models/r10.4.1_6mer.tsv \
  --ref-fasta $REF \
  --device cuda \
  > $RESULTS/crossattn.bam 2> $RESULTS/crossattn.log

# --- Vergleich ---
echo "=== VANILLA ===" > $RESULTS/comparison.txt
samtools stats $RESULTS/vanilla.bam | grep ^SN >> $RESULTS/comparison.txt
echo "" >> $RESULTS/comparison.txt
echo "=== CROSS-ATTENTION ===" >> $RESULTS/comparison.txt
samtools stats $RESULTS/crossattn.bam | grep ^SN >> $RESULTS/comparison.txt

echo "=== Ergebnis D5 (Human HG001) ==="
cat $RESULTS/comparison.txt
SLURM_EOF

sbatch -D $WORK $WORK/eval_d5.sh

# ============================================================================
# 11) Evaluate mit Bonito's eingebautem Evaluator (auf Trainingsdaten)
# ============================================================================
cat > $WORK/eval_builtin.sh << 'SLURM_EOF'
#!/bin/bash
#SBATCH --job-name=bonito-eval
#SBATCH --partition=gpu_part
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

export WORK=/mnt/galactica/$USER/bonitohash
cd $WORK/BonitoHash
source $WORK/venv/bin/activate

DATA=$WORK/data
RESULTS=$WORK/results/eval_builtin
mkdir -p $RESULTS

echo "=== Vanilla Model Eval ==="
bonito evaluate $WORK/runs/vanilla \
  --directory $DATA/training/example_data_dna_r10.4.1_v0 \
  --dataset valid \
  --chunks 1024 \
  --device cuda \
  --output_dir $RESULTS/vanilla \
  2>&1 | tee $RESULTS/vanilla_eval.txt

echo ""
echo "=== Cross-Attention Model Eval ==="
bonito evaluate $WORK/runs/crossattn \
  --directory $DATA/training/example_data_dna_r10.4.1_v0 \
  --dataset valid \
  --chunks 1024 \
  --device cuda \
  --output_dir $RESULTS/crossattn \
  2>&1 | tee $RESULTS/crossattn_eval.txt

# Vergleich
echo ""
echo "========================================="
echo "  VERGLEICH: Vanilla vs Cross-Attention"
echo "========================================="
echo "--- Vanilla ---"
grep -E "accuracy|sub-rate|ins-rate|del-rate" $RESULTS/vanilla_eval.txt
echo ""
echo "--- Cross-Attention ---"
grep -E "accuracy|sub-rate|ins-rate|del-rate" $RESULTS/crossattn_eval.txt
SLURM_EOF

sbatch -D $WORK $WORK/eval_builtin.sh

# ============================================================================
# NÜTZLICHE BEFEHLE
# ============================================================================
# Job-Status:            squeue -u $USER
# Job-Details:           scontrol show job JOBID
# Job abbrechen:         scancel JOBID
# Alle Jobs abbrechen:   scancel -u $USER
# GPU-Auslastung prüfen: srun -p gpu_part --gres gpu:1 --pty nvidia-smi
# Logs anschauen:        tail -f $WORK/bonito-vanilla_JOBID.out
# Interaktive GPU-Shell: srun -p gpu_part --gres gpu:1 -c 8 --mem=32GB -D $WORK --pty bash -l
# Disk-Usage:            du -sh $WORK/*

# ============================================================================
# QUICK-TEST: Alles lokal auf safari-gpu0 kurz testen (OHNE Slurm)
# ============================================================================
# ssh NETHZ@safari-gpu0.ethz.ch
# cd /mnt/galactica/$USER/bonitohash/BonitoHash
# source /mnt/galactica/$USER/bonitohash/venv/bin/activate
#
# export DATA=/mnt/galactica/$USER/bonitohash/data
#
# # Schneller Trainingstest (wenige Chunks, 1 Epoche)
# bonito train /tmp/test_vanilla \
#   --pretrained dna_r10.4.1_e8.2_400bps_sup@v5.0.0 \
#   --directory $DATA/training/example_data_dna_r10.4.1_v0 \
#   --epochs 1 --batch 16 --chunks 256 --device cuda -f
#
# bonito train /tmp/test_crossattn \
#   --config bonito/models/configs/dna_r10.4.1@v5.0_crossattn.toml \
#   --directory $DATA/training/example_data_dna_r10.4.1_v0 \
#   --kmer-model $DATA/kmer_models/r10.4.1_6mer.tsv \
#   --epochs 1 --batch 16 --chunks 256 --device cuda -f
