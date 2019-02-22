#  OpenKiwi: Open-Source Machine Translation Quality Estimation
#  Copyright (C) 2019 Unbabel <openkiwi@unbabel.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import numpy as np


def word_dist(word, ref_word, src_idx, trg_idx):
    """
    Distance metric for words including aligning to None
    """
    if word is None:
        # deletion
        cost = 1
    elif ref_word is None:
        # insertion
        cost = 1
    elif word == ref_word:
        # Exact match
        cost = 0
    elif word.lower() == ref_word.lower():
        # Case normalized match
        cost = 0
    # elif
    # TODO: small lexical change
    else:
        # no match
        cost = 1
    return cost


def alignments_from_filled_lists(
    src, trg, aligned_src, aligned_trg, filler=None
):
    """
    Transforms aligned lists into a list of aligment index pairs similar to
    those used in MT. It wll include alignements to nothing in both directions
    (dels/inserts)
    """

    assert len(aligned_src) == len(aligned_trg), "Token lists must be aligned"

    alignments = []
    src_idx = 0
    trg_idx = 0
    for al_src, al_trg in zip(aligned_src, aligned_trg):
        if al_src == filler:
            # Inserted
            alignments.append(((src_idx - 1, src_idx), trg_idx))
            trg_idx += 1
        elif al_trg == filler:
            # Deleted
            alignments.append((src_idx, (trg_idx - 1, trg_idx)))
            src_idx += 1
        else:
            # Replaced/ Match
            alignments.append((src_idx, trg_idx))
            src_idx += 1
            trg_idx += 1

    # TODO: map del+ins of same word to a shift. Use shortest distance
    # to assign. Replace insertion, deletion by shift.

    # Compute the edits with respect to the reference.
    edits = []
    for al_pair in alignments:
        if not isinstance(al_pair[1], int):
            # Insertion
            edits.append((al_pair[0], (src[al_pair[0]], None)))

        elif not isinstance(al_pair[0], int):
            # Deletion
            edits.append((al_pair[0], (None, trg[al_pair[1]])))

        elif src[al_pair[0]] != trg[al_pair[1]]:
            # Edition
            edits.append((al_pair[0], (src[al_pair[0]], trg[al_pair[1]])))

    return alignments, edits


def align_lists(
    list_src, list_trg, filler=None, token_dist=None, align_out='filler'
):
    """
    Aligns two lists of iterables using dynamic programming.
    """

    # Be sure the filler is unused
    assert all([item != filler for item in list_src + list_trg]), (
        "Filler set to %s. It can no be present in the input strings" % filler
    )

    # Scoring function
    if token_dist is None:
        token_dist = word_dist

    # INITIALIZATION
    nr_tokens_src = len(list_src)
    nr_tokens_trg = len(list_trg)

    # A matrix best_par_dist is created to save the optimal dynamic programming
    # scores for every pair of substrings up to that point.
    best_par_dist = np.zeros([nr_tokens_trg + 1, nr_tokens_src + 1])
    backpointers = np.NaN * np.ones(best_par_dist.shape)

    # Initialize the first row and column with deletion costs
    ds = word_dist('_', None, 1, 1)
    best_par_dist[1 : nr_tokens_trg + 1, 0] = ds * np.arange(
        2, nr_tokens_trg + 2
    )
    best_par_dist[0, 1 : nr_tokens_src + 1] = ds * np.arange(
        2, nr_tokens_src + 2
    )

    # Compute distances matrix and simultaeneously solve the dynamic
    # programming problem
    for src_idx in range(1, nr_tokens_src + 1):
        for trg_idx in range(1, nr_tokens_trg + 1):
            # scores for each possibility
            match = best_par_dist[trg_idx - 1, src_idx - 1] + word_dist(
                list_src[src_idx - 1], list_trg[trg_idx - 1], src_idx, trg_idx
            )
            insert = best_par_dist[trg_idx - 1, src_idx] + word_dist(
                None, list_trg[trg_idx - 1], src_idx, trg_idx
            )
            delete = best_par_dist[trg_idx, src_idx - 1] + word_dist(
                list_src[src_idx - 1], None, src_idx, trg_idx
            )
            # max and argmax
            best_par_dist[trg_idx, src_idx] = np.min([match, insert, delete])
            backpointers[trg_idx, src_idx] = np.argmin([match, insert, delete])
    # Get best distance
    best_dist = best_par_dist[nr_tokens_trg, nr_tokens_src]

    return best_dist, nr_tokens_src, best_dist / nr_tokens_src

    # Do the Traceback to create the alignment. To create the traceback,
    # reconstruct the path from position best_par_dist[nr_tokens_trg+1,
    # nr_tokens_src+1] to best_par_dist[1, 1] that led to the highest score.
    trg_idx = nr_tokens_trg
    src_idx = nr_tokens_src
    list_src_aligned = []
    list_trg_aligned = []
    # Go backwards until start of matrix
    while trg_idx > 0 and src_idx > 0:
        if backpointers[trg_idx, src_idx] == 0:
            list_src_aligned.append(list_src[src_idx - 1])
            list_trg_aligned.append(list_trg[trg_idx - 1])
            trg_idx = trg_idx - 1
            src_idx = src_idx - 1

        elif backpointers[trg_idx, src_idx] == 2:
            list_src_aligned.append(list_src[src_idx - 1])
            list_trg_aligned.append(filler)
            src_idx = src_idx - 1

        elif backpointers[trg_idx, src_idx] == 1:
            list_src_aligned.append(filler)
            list_trg_aligned.append(list_trg[trg_idx - 1])
            trg_idx = trg_idx - 1

    # At this point either trg_idx or src_idx (or both) should be one.
    # Finish to the ends by adding gaps and the rest of the
    # remaining string until both counters, trg_idx and src_idx, are equal to
    # one.
    if src_idx > 0:
        while src_idx > 0:
            list_src_aligned.append(list_src[src_idx - 1])
            list_trg_aligned.append(filler)
            src_idx = src_idx - 1
    elif trg_idx > 0:
        while trg_idx > 0:
            list_src_aligned.append(filler)
            list_trg_aligned.append(list_trg[trg_idx - 1])
            trg_idx = trg_idx - 1

    # Reverse lists
    list_src_aligned.reverse()
    list_trg_aligned.reverse()

    # Format output
    if align_out == 'pairs':
        result = alignments_from_filled_lists(
            list_src,
            list_trg,
            list_src_aligned,
            list_trg_aligned,
            filler=filler,
        )
    elif align_out == 'filler':
        result = list_src_aligned, list_trg_aligned
    else:
        raise Exception("Uknown alignment output %s" % align_out)
    print(result)
    return best_dist, nr_tokens_src, best_dist / nr_tokens_src


if __name__ == '__main__':
    # ref_file = 'data/WMT17/sentence_level/en_de/train.pe'
    # hyp_file = 'data/WMT17/sentence_level/en_de/dev.mt'
    # ter_file = 'data/WMT17/sentence_level/en_de/dev.hter.ter'

    ref_file = 'data/WMT18/sentence_level/en_de/dev.smt.pe'
    hyp_file = 'data/WMT18/sentence_level/en_de/dev.smt.mt'
    ter_file = 'data/WMT18/sentence_level/en_de/dev.smt.hter'
    tercom_file = 'data/WMT18/sentence_level/en_de/dev.smt.hter'

    with open(ref_file) as fr:
        ref = [line.strip().split() for line in fr]
    # with open(hyp_file) as fh:
    #     hyp = [line.strip().split() for line in fh]
    # with open(ter_file) as f:
    #     ter = [line.strip().split()[1] for line in f]

    import editdistance

    def tercom(r, h):
        return editdistance.eval(r, h) / len(r)

    eds = [
        (tercom(r, h), i, j)
        for i, r in enumerate(ref)
        for j, h in enumerate(ref)
        if i < j and tercom(r, h) <= 0.4
    ]
    with open('data/WMT17/sentence_level/en_de/train.tmp', 'w') as f:
        for ed in sorted(eds):
            f.write('{}\t{}\t{}\n'.format(*ed))

    # for r, h in zip(ref, hyp)])
    # for i, (r, h) in enumerate(zip(ref, hyp)):
    #     ed = editdistance.eval(r, h)
    #     n, t, l = align_lists(r, h, align_out='pairs')
    #     if ed != n:
    #         f.write('{}: {} != {} (ter: {})\n'.format(i, ed, n, ter[i]))
