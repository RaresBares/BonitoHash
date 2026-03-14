"""
Bonito Basecaller
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from time import perf_counter
from functools import partial
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.nn import fuse_bn_
from bonito.aligner import align_map, Aligner
from bonito.reader import read_chunks, Reader
from bonito.io import CTCWriter, Writer, biofmt
from bonito.cli.download import Downloader, models, __models_dir__
from bonito.multiprocessing import process_cancel, process_itemmap
from bonito.util import column_to_set, load_symbol, load_model, init, tqdm_environ


def main(args):

    init(args.seed, args.device)

    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)

    fmt = biofmt(aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    if args.model_directory in models and not (__models_dir__ / args.model_directory).exists():
        sys.stderr.write("> downloading model\n")
        Downloader(__models_dir__).download(args.model_directory)

    sys.stderr.write(f"> loading model {args.model_directory}\n")
    try:
        model = load_model(
            args.model_directory,
            args.device,
            weights=args.weights if args.weights > 0 else None,
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
            use_koi=True,
        )
        model = model.apply(fuse_bn_)
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        for model in sorted(models): sys.stderr.write(f" - {model}\n")
        exit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model.config['basecaller']}\n")

    basecall = load_symbol(args.model_directory, "basecall")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset=args.mm2_preset)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    if fmt.name != 'fastq':
        groups, num_reads = reader.get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=8, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []
        num_reads = None

    reads = reader.get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        do_trim=not args.no_trim,
        scaling_strategy=model.config.get("scaling"),
        norm_params=(model.config.get("standardisation")
                     if (model.config.get("scaling") and
                         model.config.get("scaling").get("strategy") == "pa")
                     else model.config.get("normalisation")
                     ),
        cancel=process_cancel()
    )

    if args.verbose:
        sys.stderr.write(f"> read scaling: {model.config.get('scaling')}\n")
    
    if args.max_reads:
        reads = take(reads, args.max_reads)
        if num_reads is not None:
            num_reads = min(num_reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"]
            )
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = Writer

    ref_provider = None
    if args.rawhash_paf and args.kmer_model and args.ref_fasta:
        from bonito.rawhash.paf_parser import parse_paf
        from bonito.rawhash.kmer_model import KmerModel
        import pysam
        import torch

        sys.stderr.write("> loading RawHash mappings\n")
        mappings = parse_paf(args.rawhash_paf)
        kmer_model_inst = KmerModel(args.kmer_model)
        ref_fasta = pysam.FastaFile(args.ref_fasta)

        _complement = str.maketrans('ACGTacgt', 'TGCAtgca')
        def _reverse_complement(seq):
            return seq.translate(_complement)[::-1]

        def ref_provider(read):
            read_id = read.read_id if hasattr(read, 'read_id') else None
            if read_id is None or read_id not in mappings:
                return None
            m = mappings[read_id]
            ref_seq = ref_fasta.fetch(m.target_name, m.target_start, m.target_end)
            if m.strand == '-':
                ref_seq = _reverse_complement(ref_seq)
            base_map = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0,
                        'a': 1, 'c': 2, 'g': 3, 't': 4, 'n': 0}
            bases = np.array([base_map.get(b, 0) for b in ref_seq], dtype=np.uint8)
            kmer_ids = kmer_model_inst.bases_to_kmer_ids(bases)
            expected_signals = kmer_model_inst.get_expected_signal(kmer_ids)
            return (
                torch.from_numpy(kmer_ids).unsqueeze(0).long(),
                torch.from_numpy(expected_signals).unsqueeze(0).float(),
            )

        sys.stderr.write(f"> loaded {len(mappings)} RawHash mappings\n")

    results = basecall(
        model, reads, reverse=args.revcomp, rna=args.rna,
        batchsize=model.config["basecaller"]["batchsize"],
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"],
        ref_provider=ref_provider,
    )

    if aligner:
        results = align_map(aligner, results, n_thread=args.alignment_threads)

    writer_kwargs = {'aligner': aligner,
                     'group_key': args.model_directory,
                     'ref_fn': args.reference,
                     'groups': groups,
                     'min_qscore': args.min_qscore}
    if args.save_ctc:
        writer_kwargs['rna'] = args.rna
        writer_kwargs['min_accuracy'] = args.min_accuracy_save_ctc
        
    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False,
                       total=num_reads, smoothing=0, ascii=True, ncols=100,
                       **tqdm_environ()),
        **writer_kwargs)

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--min-qscore", default=0, type=int)
    parser.add_argument("--min-accuracy-save-ctc", default=0.99, type=float)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument("--mm2-preset", default='lr:hq', type=str)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument("--rawhash-paf", default=None, type=str,
                        help="PAF file from RawHash mapping for reference-guided basecalling")
    parser.add_argument("--kmer-model", default=None, type=str,
                        help="Path to ONT k-mer pore model TSV")
    parser.add_argument("--ref-fasta", default=None, type=str,
                        help="Reference FASTA for extracting sequences from RawHash mappings")
    return parser
