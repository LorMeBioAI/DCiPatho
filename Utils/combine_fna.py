import os

from Utils.cal_utils import readfq


# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中
# 输入文件夹路径、空文件列表[]
# 返回 文件列表Filelist,包含文件名（完整路径）

def get_filelist(dir, Filelist):
    if os.path.isfile(dir):
        Filelist.append(dir)

        # # 若只是要返回文件文，使用这个

        # Filelist.append(os.path.basename(dir))

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码

            # if s == "xxx":

            # continue
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)

    return Filelist


# from a path to one fna file
def combine(fastapath, out_file):
    list = get_filelist(fastapath, [])
    print('num of lists:', len(list))
    for s in list:
        seq_names = []
        seqs = []
        i = 0
        # print(s)
        fp = open(s)
        for name, seq, _ in readfq(fp):
            print('name111', name)
            # print('name 0', name[0])
            seq_names.append(name.strip("'"))
            seqs.append(seq)
            i += 1
        # connection of seqs
        allseqs = ''.join(seqs)
        # save first name

        # print('seq_names[0]:', seq_names[0])
        with open(out_file, 'a') as f:
            f.write('>%s \n' % str(seq_names[0]))

            f.write('%s \n' % str(allseqs))
        print("%s fasta file read %s sequences" % (s, i))


def get_name(fasta_name):
    seq_names = []
    seqs = []
    seq_lens = []
    e = open(fasta_name)
    for name, seq, _ in readfq(e):
        seq_names.append(name[0])
        seq_lens.append(len(seq))
    return seq_lens
