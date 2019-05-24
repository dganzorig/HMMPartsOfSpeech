#!/usr/bin/env python3.

import numpy as np
import random
import math
import argparse as ap
from collections import defaultdict
from enum import Enum

verbose = True
output_file = None
verbose_out = None


class HMM:

    def __init__(self, _n, _alphabet, _pi, _states):
        self._n = _n
        self._alphabet = _alphabet
        self._alphabet_dictionary = {}
        for index, letter in enumerate(self._alphabet):
            self._alphabet_dictionary[letter] = index
        assert (len(_pi) == self._n)
        self._pi = _pi
        assert (len(_states) == self._n)
        self._states = _states

    @property
    def n(self):
        return self._n

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def pi(self):
        return self._pi

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, _states):
        assert(len(_states) == self._n)
        self._states = _states

    def forward_trellis(self, sentence):
        trellis = [[0 for _ in range(len(sentence)+1)] for _ in range(self._n)]
        for i in range(self._n):
            trellis[i][0] = self._pi[i]
        for t in range(1, len(sentence) + 1):
            for i in range(self._n):
                alpha_i = 0
                for j in range(self._n):
                    alpha_j = trellis[j][t-1]
                    a_j = self._states[j].A[i]
                    b_j = self._states[j].B[sentence[t-1]]
                    alpha_i += alpha_j * a_j * b_j
                trellis[i][t] = alpha_i
        total = 0
        for i in range(self._n):
            total += trellis[i][len(sentence)]
        return trellis, total

    def backward_trellis(self, sentence):
        trellis = [[0 for _ in range(len(sentence) + 1)] for _ in range(self._n)]
        for i in range(self._n):
            trellis[i][len(sentence)] = 1
        for t in range(len(sentence)-1, -1, -1):
            for i in range(self._n):
                beta_i = 0
                for j in range(self._n):
                    beta_j = trellis[j][t+1]
                    a_j = self._states[i].A[j]
                    b_j = self._states[i].B[sentence[t]]
                    beta_i += beta_j * a_j * b_j
                trellis[i][t] = beta_i
        total = 0
        for i in range(self._n):
            total += self._pi[i] * trellis[i][0]
        return trellis, total

    def get_soft_counts(self, in_file):
        total_sentences = 0
        initial_soft_counts = defaultdict(lambda: 0)
        soft_counts = defaultdict(lambda: 0)
        with open(in_file, 'r') as ins:
            for line in ins:
                sentence = line.split(' ')
                sentence = normalize_sentence(sentence)
                total_sentences += 1
                f_trellis, f_total = self.forward_trellis(sentence)
                b_trellis, b_total = self.backward_trellis(sentence)
                for i, word in enumerate(sentence):
                    sum_check = 0
                    for state_1 in range(0, self._n):
                        for state_2 in range(0, self._n):
                            key = (word, state_1, state_2)
                            alpha = f_trellis[state_1][i]
                            a = self._states[state_1].A[state_2]
                            b = self._states[state_1].B[word]
                            beta = b_trellis[state_2][i+1]
                            proportion = alpha * a * b * beta / f_total
                            if i == 0:
                                initial_soft_counts[key] += proportion
                            soft_counts[key] += proportion
                            sum_check += proportion
        return initial_soft_counts, total_sentences, soft_counts

    def viterbi(self, word, out):
        if out:
            out.write('---------------------------------\n')
            out.write('-----     Viterbi Path      -----\n')
            out.write('---------------------------------\n')
            out.write('Word: {}\n\n'.format(word))
        delta = [[0 for _ in range(len(word) + 1)] for _ in range(self._n)]
        phi = [[0 for _ in range(len(word) + 1)] for _ in range(self._n)]
        max_pi_state = max(enumerate(self._pi), key=lambda x: x[1])[0]
        for i in range(self._n):
            delta[i][0] = self.pi[i]
            phi[i][0] = max_pi_state
            if out:
                out.write('Delta[0] of state {}: {}\n'.format(i, delta[i][0]))
        if out:
            out.write('\n')
        for t in range(1, len(word) + 1):
            if out:
                out.write('Time t + 1: {} \t Letter: {}\n'.format(t, word[t-1]))
            for i in range(self._n):
                if out:
                    out.write('at state {}:\n'.format(i))
                prevs = []
                for j in range(self._n):
                    prev_prob = delta[j][t-1]
                    a_ji = self._states[j].A[i]
                    b_jo = self._states[j].B[word[t-1]]
                    current_prob = prev_prob * a_ji * b_jo
                    candidate_tuple = (j, current_prob)
                    prevs.append(candidate_tuple)
                    if out:
                        out.write('\tfrom-state {}: {}\n'.format(j, current_prob))
                max_tuple = max(prevs, key=lambda x: x[1])
                phi[i][t] = max_tuple[0]
                delta[i][t] = max_tuple[1]
                if out:
                    out.write('best state to come from is: {} (at {})\n\n'.format(max_tuple[0], max_tuple[1]))
            if out:
                out.write('\n')
        finals = []
        if out:
            out.write('Final probabilities:\n')
        for i in range(self._n):
            final_prob = delta[i][len(word)]
            candidate_tuple = (i, final_prob)
            finals.append(candidate_tuple)
            out.write('\tState {}: {}\n'.format(i, final_prob))
        highest_final = max(finals, key=lambda x: x[1])
        best_index = highest_final[0]
        backward_path = [best_index]
        for t in range(len(word) - 1, -1, -1):  # stops after i = 0
            left_best_index = phi[best_index][t]
            backward_path.append(left_best_index)
            best_index = left_best_index
        forward_path = backward_path[::-1]
        if out:
            out.write('\nViterbi path:\n')
            for index, state in enumerate(forward_path):
                out.write('time: {}\tstate: {}\n'.format(index, state))
            out.write('---------------------------------------------')
        return forward_path


class State:

    def __init__(self, _id, _A, _B):
        self._id = id
        self._A = _A
        self._B = _B

    @property
    def id(self):
        return self._id

    @property
    def A(self):
        return self._A

    @A.setter
    def A(self, _B):
        self._B = _B

    @property
    def B(self):
        return self._B

    @B.setter
    def B(self, _B):
        self._B = _B


class PartOfSpeech(Enum):
    PUNC = 1
    ART = 2
    ADJ = 3
    VERB = 4
    NOUN = 5
    AUX = 6
    PRONOUN = 7
    NEG = 8
    PREP = 9
    WH_WORD = 10


def generate_distribution(n):
    a = np.random.random(n)
    a /= a.sum()
    # round out last entry so sum() == 1:
    a[len(a) - 1] = 1 - sum(a[:-1])
    return a


def choose_random(a):
    x = random.uniform(0, 1)
    probability_accumulation = 0
    counter = 0
    for probability in a:
        probability_accumulation += probability
        if x <= probability_accumulation:
            break
        counter += 1
    return counter


def normalize_word(word):
    new_word = word.strip()
    # new_word = re.sub('[^a-zA-Z]+', '', raw_line)
    return new_word


def normalize_sentence(sentence):
    new_sentence = []
    for word in sentence:
        new_word = normalize_word(word)
        new_sentence.append(new_word)
    return new_sentence


def generate_random_hmm(n, alphabet):
    states = []
    for i in range(n):
        current_A = generate_distribution(n)
        b_probabilities = generate_distribution(len(alphabet))
        current_B = {}
        for index, letter_probability in enumerate(b_probabilities):
            current_B[alphabet[index]] = letter_probability
        current_state = State(i, current_A, current_B)
        states.append(current_state)
    pi = generate_distribution(n)
    hmm = HMM(n, alphabet, pi, states)
    return hmm


def total_probability(in_file, hmm):
    plog_sum = 0
    with open(in_file, 'r') as ins:
        for line in ins:
            sentence = line.split(' ')
            sentence = normalize_sentence(sentence)
            _, forward_probability = hmm.forward_trellis(sentence)
            _, backward_probability = hmm.backward_trellis(sentence)
            current_plog = math.log2(1.0 / forward_probability)
            plog_sum += current_plog
    return plog_sum


def output_trellis(trellis, is_alpha):
    verbose_out.write('Alpha:\n' if is_alpha else 'Beta:\n')
    for t in range(len(trellis[0])):
        verbose_out.write('Time\t{}\t'.format(t))
        for i in range(len(trellis)):
            value = round(trellis[i][t], ndigits=12)
            verbose_out.write('State {}:'.format(i).ljust(10) + '{}'.format(str(value).ljust(14)))
            verbose_out.write('\t')
        verbose_out.write('\n')
    verbose_out.write('\n')


def output_soft_count_table(table, alphabet, n, out):
    sum = 0
    for letter in alphabet:
        for i in range(0, n):
            for j in range(0, n):
                key = (letter, i, j)
                probability = table[key]
                out.write('{}{}{}{}\n'.format(str(letter).ljust(5), str(i).ljust(5),
                                              str(j).ljust(5), probability))
                sum += probability
    out.write('Sum: {}\n\n'.format(sum))


def new_pi(table, alphabet, n, z, i):
    sum = 0
    for word in alphabet:
        for j in range(0, n):
            key = (word, i, j)
            soft_count = table[key]
            sum += soft_count
    pi = sum / z
    return pi


def new_a(table, alphabet, n, i, j):
    ij_sum = 0
    for word in alphabet:
        key = (word, i, j)
        soft_count = table[key]
        ij_sum += soft_count

    i_sum = 0
    for word in alphabet:
        for k in range(0, n):
            key = (word, i, k)
            soft_count = table[key]
            i_sum += soft_count

    a = ij_sum / i_sum
    return a


def new_b(table, alphabet, n, i, l):
    j_sum = 0
    for j in range(0, n):
        key = (l, i, j)
        soft_count = table[key]
        j_sum += soft_count

    mj_sum = 0
    for word in alphabet:
        for j in range(0, n):
            key = (word, i, j)
            soft_count = table[key]
            mj_sum += soft_count

    b = j_sum / mj_sum
    return b


def local_optimum(hmm, n, alphabet, in_file):
    global verbose_out
    probability = 0
    for i in range(400):
        initial_soft_counts, total_words, soft_counts = hmm.get_soft_counts(in_file)

        next_pi = []
        next_states = []

        if verbose:
            verbose_out.write('---------------------------------------\n')
            verbose_out.write('              Maximization             \n')
            verbose_out.write('---------------------------------------\n')

        # Generate the 'n' new states and pi distribution
        for i in range(0, n):
            if verbose:
                verbose_out.write('Recalculations for state: {}\n\n'.format(i))

            # Update pi
            this_pi = new_pi(initial_soft_counts, alphabet, n, total_words, i)
            next_pi.append(this_pi)
            if verbose:
                verbose_out.write('\tNext pi: {} (prev. {})\n\n'.format(this_pi, hmm.pi[i]))

            # Generate new A
            A_i = []
            for j in range(0, n):
                next_a = new_a(soft_counts, alphabet, n, i, j)
                A_i.append(next_a)
                if verbose:
                    verbose_out.write('\tA to state {}: {} (prev. {})\n'.format(j, next_a, hmm.states[i].A[j]))
            if verbose:
                verbose_out.write('\n')

            # Generate new B
            B_i = {}
            for letter in alphabet:
                next_b = new_b(soft_counts, alphabet, n, i, letter)
                B_i[letter] = next_b
                if verbose:
                    verbose_out.write(
                        '\tB for letter {}: {} (prev. {})\n'.format(letter, next_b, hmm.states[i].B[letter]))
            if verbose:
                verbose_out.write('\n')

            new_state = State(i, A_i, B_i)
            next_states.append(new_state)

        # update HMM
        hmm = HMM(n, alphabet, next_pi, next_states)
        probability = total_probability(in_file, hmm)

    return hmm, probability


def string_to_part(part_str):
    switcher = {
        'punc': PartOfSpeech.PUNC,
        'art': PartOfSpeech.ART,
        'adj': PartOfSpeech.ADJ,
        'verb': PartOfSpeech.VERB,
        'noun': PartOfSpeech.NOUN,
        'aux': PartOfSpeech.AUX,
        'pronoun': PartOfSpeech.PRONOUN,
        'neg': PartOfSpeech.NEG,
        'prep': PartOfSpeech.PREP,
        'wh-word': PartOfSpeech.WH_WORD
    }
    part_of_speech = switcher[part_str]
    return part_of_speech


def get_corpus_alphabet(file_name):
    alphabet = []
    alphabet_dictionary = {}  # maps alphabet to part of speech
    with open(file_name) as ins:
        for line in ins:
            # assumes entries are in form: word (space) part_of_speech
            entries = line.split(' ')
            word = normalize_word(entries[0])
            part_of_speech = string_to_part(normalize_word(entries[1]))
            alphabet.append(word)
            alphabet_dictionary[word] = part_of_speech

    return alphabet, alphabet_dictionary


def calculate_purity_for_state(B, alphabet, alphabet_dictionary):
    counts = {}
    for word in alphabet:
        if B[word] >= 0.001:
            part_of_speech = alphabet_dictionary[word]
            if part_of_speech in counts:
                counts[part_of_speech] = counts[part_of_speech] + 1
            else:
                counts[part_of_speech] = 1
    max_part_of_speech = max(counts.items(), key=lambda k: k[1])
    # count of max part / total words > 0.001
    purity = counts[max_part_of_speech[0]] / sum(counts.values())
    return purity


def get_average_purity(hmm, alphabet_dictionary):
    purity_list = []
    for state in hmm.states:
        purity = calculate_purity_for_state(state.B, hmm.alphabet, alphabet_dictionary)
        purity_list.append(purity)
    average_purity = sum(purity_list) / len(purity_list)
    return average_purity


def run_purity_simulations(alphabet, alphabet_dictionary, data_file, out):
    purity_calculation_list = []
    for i in range(3, 11):
        for j in range(3):
            hmm = generate_random_hmm(i, alphabet)
            hmm, probability = local_optimum(hmm, i, alphabet, data_file)
            average_purity = get_average_purity(hmm, alphabet_dictionary)
            item = (i, average_purity)
            purity_calculation_list.append(item)
            if j == 0:
                output_hmm(hmm, out)
    return purity_calculation_list


def output_hmm(hmm, out):
    out.write('---------------------------------------------\n')
    out.write('-                     HMM                   -\n')
    out.write('---------------------------------------------\n')
    for i in range(len(hmm.states)):
        current_A = hmm.states[i].A
        out.write('State {}\n'.format(i))
        out.write('Transitions\n')
        for index, state_probability in enumerate(current_A):
            out.write('\tTo state\t{}\t{}\n'.format(index, state_probability))
        out.write('\tTotal: {}\n'.format(sum(current_A)))
        current_B = hmm.states[i].B
        out.write('\nEmission probabilities\n')
        sorted_B = sorted(current_B.items(), key=lambda kv: kv[1], reverse=True)
        for key_value in sorted_B:
            if key_value[1] >= 0.001:
                out.write('\tWord: {} {}\n'.format(key_value[0].ljust(30),
                                                   str(key_value[1]).ljust(15)))
        out.write('\tTotal: {}\n\n'.format(sum(current_B.values())))
    pi = hmm.pi
    out.write('---------------------------\n')
    out.write('Pi:\n')
    for index, state_probability in enumerate(pi):
        out.write('State\t{}\t{}\n'.format(index, state_probability))
    out.write('---------------------------\n\n\n')


def output_purities(purity_calculations, out):
    first_header = 'Number of States'
    second_header = 'Average Purity'
    out.write('{} | {}\n'.format(first_header, second_header))
    out.write('----------------------------------------')
    out.write('\n')
    for purity_calculation in purity_calculations:
        out.write(str(purity_calculation[0]).ljust(len(first_header) + 5))
        out.write(str(purity_calculation[1]).ljust(len(second_header)))
        out.write('\n')
    out.write('\n')


def main():
    parser = ap.ArgumentParser()
    parser.add_argument('--v', help='Show debugging prints', action='store_true', default=False)
    parser.add_argument('--i', help='Input file of words', default='words.txt')
    parser.add_argument('--d', help='Input file of toy data', default='toy_data.txt')
    parser.add_argument('--o', help='Output file', default='output.txt')
    args, _ = parser.parse_known_args()

    global verbose
    verbose = args.v
    if verbose:
        global verbose_out
        verbose_out = open('verbose.txt', 'w')
    words_file = args.i
    data_file = args.d
    global output_file
    output_file = open(args.o, 'w')

    alphabet, alphabet_dictionary = get_corpus_alphabet(words_file)
    purity_calculation_list = run_purity_simulations(alphabet, alphabet_dictionary, data_file, output_file)
    output_purities(purity_calculation_list, output_file)

    if verbose:
        verbose_out.close()
    output_file.close()


if __name__ == '__main__':
    main()
