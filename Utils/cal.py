import itertools
import multiprocessing as mp
import time
from multiprocessing import Manager

import numpy as np

from Utils.cal_utils import count_kmers, readfq, mer2bits, get_rc


def compute_kmer_inds(ks):
    ''' Get the indeces of each canonical kmer in the kmer count vectors
    '''
    kmer_list = []
    kmer_inds = {k: {} for k in ks}
    kmer_count_lens = {k: 0 for k in ks}

    alphabet = 'ACGT'
    for k in ks:
        all_kmers = [''.join(kmer) for kmer in itertools.product(alphabet, repeat=k)]
        # ��������kmers����� 4��k�η���
        all_kmers.sort()
        # ���� ��ascii���С��
        ind = 0
        for kmer in all_kmers:
            bit_mer = mer2bits(kmer)
            rc_bit_mer = mer2bits(get_rc(kmer))
            if rc_bit_mer in kmer_inds[k]:
                kmer_inds[k][bit_mer] = kmer_inds[k][rc_bit_mer]
            else:
                kmer_list.append(kmer)
                kmer_inds[k][bit_mer] = ind
                kmer_count_lens[k] += 1
                ind += 1
    # return kmer_inds, kmer_count_lens, kmer_list
    return kmer_inds, kmer_count_lens


def get_seq_lengths(infile):
    ''' Read in all the fasta entries,
        return arrays of the headers, and sequence lengths
    '''
    sequence_names = []
    sequence_lengths = []
    seqs = []
    fp = open(infile)
    for name, seq, _ in readfq(fp):
        sequence_names.append(name)
        sequence_lengths.append(len(seq))
        seqs.append(seq)
    fp.close()

    # print('len of sequence_lengths', len(sequence_names), len(sequence_lengths))

    return sequence_names, sequence_lengths, seqs


def get_seq(infile):
    seq_names = []
    seqs = []
    fp = open(infile)
    i = 0

    for name, seq, _ in readfq(fp):
        seq_names.append(name)

        seqs.append(seq)
        # if len(seqs) == 63911:
        #     break

        i += 1
        # if i % 100000 == 0:
        # if i < 100000:
    print("Read {} sequences".format(i))
    return seq_names, seqs


def get_num_frags(seq_lengths, length, coverage=5):
    ''' Compute how many fragments to generate
    '''
    # filter out sequences that are significantly shorter than the length ���˵����Զ����������е�����
    # filtered_seq_lengths = [l for l in seq_lengths if l > 0.85*length]
    # filtered_seq_lengths =
    tot_seq_len = sum(seq_lengths)

    num_frags_for_cov = int(np.ceil(tot_seq_len * coverage / float(length)))
    num_frags = min(90000, num_frags_for_cov)
    # num_frags = (tot_seq_len)
    return num_frags


# def get_start_inds(seq_names, seq_lengths, num_frags, length):
def get_start_inds(seq_names):
    ''' Randomly simulate fragments of a specific length from the sequences ���ģ���������ض����ȵ�Ƭ��
    '''
    # filter out sequences that are significantly shorter than the length
    # filtered_seq_names = [seq_names[i] for i,v in enumerate(seq_lengths) if v > 0.05*length]
    # filtered_seq_lengths = [l for l in seq_lengths if l > 0.05*length]
    # filtered_seq_lengths = seq_lengths
    # tot_seq_len = sum(seq_lengths)
    # length_fractions = [float(l) / float(tot_seq_len) for l in seq_lengths]
    inds_dict = {}
    for name in seq_names:
        inds_dict[name] = [0]
    #
    # for i in range(num_frags):
    #     # choose genome
    #     # �����������
    #     seq_ind = np.random.choice(len(seq_names), p=length_fractions)
    #     seq_len = seq_lengths[seq_ind]
    #     seq_name = seq_names[seq_ind]
    #     # choose start index in the genome
    #     # if seq_len < length: # just take the whole thing
    #     inds_dict[seq_name].append(0)
    return inds_dict


def get_seqs(infile):
    ''' Create array of the sequences
    '''
    seqs = []
    fp = open(infile)
    for name, seq, _ in readfq(fp):
        seqs.append(seq)

    # start_inds = inds_dict.get(name, [])
    # for start_ind in start_inds:
    #     frag = seq[start_ind:start_ind+l]
    #     # if len(frag) < l and len(seq) > l:
    #     #     frag += seq[:l-len(frag)]
    #     seqs.append(frag)
    fp.close()
    return seqs


def cal_kmer_freqs(seq_file, num_procs, ks):
    names, seqs = get_seq(seq_file)
    time_start = time.time()
    # patho_names, patho_lengths, patho_seqs = get_seq_lengths(patho_file)
    ## for l in lens:
    # coverage=1 # TODO: make this command line option
    kmer_inds, kmer_count_lens = compute_kmer_inds(ks)
    # print('kmer_inds: /n',kmer_inds)
    # print('kmer_count_lens:/n', kmer_count_lens)
    pool = mp.Pool(num_procs)
    # ����̹���ȫ�ֱ���
    patho_list = Manager().list()
    for cur in np.arange(len(seqs)):
        patho_list.append(0)
    pool.map(count_kmers, [[ind, s, ks, kmer_inds, kmer_count_lens, patho_list] \
                           for ind, s in enumerate(seqs)])

    patho_freqs = np.array(patho_list)
    pool.close()
    time_end = time.time()
    print('costs:', int(time_end - time_start), 's')

    return names, patho_freqs


def cal_main(combined_fasta_path, num_procs, ks, freqs_file):
    names, freqs = cal_kmer_freqs(combined_fasta_path, num_procs, ks)
    np.save(freqs_file, freqs)
    return names
