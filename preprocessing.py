import numpy as np
import os
import pickle
import re
import sys
import argparse


class FileNotFoundError(OSError):
    pass


class Preprocess():
    def __init__(self, path_to_dataset, c_max_len):
        # path_to_dataset example: '././babi_original'
        self.path_to_dataset = path_to_dataset
        self.train_paths = None
        self.val_paths = None
        self.test_paths = None
        self.all_paths = None
        self._c_word_set = set()
        self._q_word_set = set()
        self._a_word_set = set()
        self._cqa_word_set = set()
        self._all_word_set = set()
        self.c_max_len = c_max_len
        self.s_max_len = 0
        self.q_max_len = 0
        self.mask_index = 0

    def set_path(self, path_to_dataset, all_paths_to_babi):
        """Set list of train, val, and test dataset paths."""
        self.path_to_dataset = path_to_dataset

        train_paths = []
        val_paths = []
        test_paths = []
        for dirpath, dirnames, filenames in os.walk(path_to_dataset):
            for filename in filenames:
                if filename.endswith('.txt'):
                    if 'train' in filename:
                        train_paths.append(os.path.join(dirpath, filename))
                    elif 'val' in filename:
                        val_paths.append(os.path.join(dirpath, filename))
                    else:
                        assert 'test' in filename
                        test_paths.append(os.path.join(dirpath, filename))
                else:
                    print("Ignored file: {}".format(filename))
        self.train_paths = sorted(train_paths)
        self.val_paths = sorted(val_paths)
        self.test_paths = sorted(test_paths)

        all_paths = []
        for dirpath, dirnames, filenames in os.walk(all_paths_to_babi):
            for filename in filenames:
                if filename.endswith('.txt'):
                    all_paths.append(os.path.join(dirpath, filename))
                else:
                    print("Ignored file: {}".format(filename))
        self.all_paths = sorted(all_paths)

    def _split_paragraphs(self, path_to_file):
        """Split into paragraphs."""
        with open(path_to_file, 'r') as f:
            babi = f.readlines()
        paragraph = []
        paragraphs = []
        alphabet = re.compile('[a-zA-Z]')
        for d in babi:
            if d.startswith('1 '):
                if paragraph:
                    paragraphs.append(paragraph)
                paragraph = []
            mark = re.search(alphabet, d).span()[0]
            paragraph.append(d[mark:])
        return paragraphs

    def _split_clqa(self, paragraphs, show_print=True):
        """For each paragraph, split into context, label, question and answer.

        Args:
            paragraphs: list of paragraphs

        Returns:
            context: list of contexts
            label: list of labels
            question: list of questions
            answer: list of answers
        """
        context = []
        label = []
        question = []
        answer = []
        for paragraph in paragraphs:
            for i, sent in enumerate(paragraph):
                if '?' in sent:
                    related_para = [para.strip().lower() for para in paragraph[:i] if '?' not in para][::-1]
                    # Get rid of tab symbol
                    related_para = [para.split('\t')[0] for para in related_para]
                    if len(related_para) > 20:
                        related_para = related_para[:20]
                    context.append(related_para)
                    label.append([i for i in range(len(related_para))])
                    q_a_ah = sent.split('\t')
                    question.append(q_a_ah[0].strip().lower())
                    answer.append(q_a_ah[1].strip().lower())
        # check
        if show_print:
            if (len(question) == len(answer)) & (len(answer) == len(context)) & (len(context) == len(label)):
                print("bAbI is well separated into question, answer, context, and label!")
                print("total: {}".format(len(label)))
            else:
                print("Something is missing! check again")
                print("the number of questions: {}".format(len(question)))
                print("the number of answers: {}".format(len(answer)))
                print("the number of contexts: {}".format(len(context)))
                print("the number of labels: {}".format(len(label)))
        return context, label, question, answer

    def split_all_clqa(self, paths, show_print=True):
        """Merge all tasks into one dataset.

        Args:
            paths: list of path to tasks

        Returns:
            contexts: list of contexts of all tasks
            labels: list of labels of all tasks
            questions: list of questions of all tasks
            answers: list of answers of all tasks
        """
        if paths is None:
            print('path is None, run set_path() first!')
        else:
            contexts = []
            labels = []
            questions = []
            answers = []
            for path in paths:
                if show_print:
                    print('=================')
                paragraphs = self._split_paragraphs(path)
                if show_print:
                    print("data: {}".format(os.path.basename(path)))
                context, label, question, answer = self._split_clqa(paragraphs, show_print=show_print)
                contexts.extend(context)
                labels.extend(label)
                questions.extend(question)
                answers.extend(answer)
            return contexts, labels, questions, answers

    def set_word_set(self, word_set_path):

        try:
            c_word_set, q_word_set, a_word_set = np.load(word_set_path)

        except Exception as e:
            # Create the word set from the training, validation, and test data
            c_word_set = set()
            q_word_set = set()
            a_word_set = set()

            # Global vocabulary across multiple datasets
            all_contexts, all_labels, all_questions, all_answers = self.split_all_clqa(
                self.all_paths, show_print=False)

            for para in all_contexts:
                for sent in para:
                    sent = sent.replace(".", " .")
                    sent = sent.replace("?", " ?")
                    sent = sent.split()
                    c_word_set.update(sent)

            for sent in all_questions:
                sent = sent.replace(".", " .")
                sent = sent.replace("?", " ?")
                sent = sent.split()
                q_word_set.update(sent)

            for answer in all_answers:
                answer = answer.split(',')
                a_word_set.update(answer)

            a_word_set.add(',')

            # Save the word set if requested
            if word_set_path is not None and isinstance(e, FileNotFoundError):
                np.save(word_set_path, (c_word_set, q_word_set, a_word_set))

        self._c_word_set = c_word_set
        self._q_word_set = q_word_set
        self._a_word_set = a_word_set
        self._cqa_word_set = c_word_set.union(q_word_set).union(a_word_set)
        self._qa_word_set = c_word_set.union(q_word_set).union(a_word_set)

    def _index_context(self, contexts):
        c_word_index = dict()
        for i, word in enumerate(self._c_word_set):
            c_word_index[word] = i+1  # index 0 for zero padding
        indexed_cs = []
        for context in contexts:
            indexed_c = []
            for sentence in context:
                sentence = sentence.replace(".", " .")
                sentence = sentence.replace("?", " ?")
                sentence = sentence.split()
                indexed_s = []
                for word in sentence:
                    indexed_s.append(c_word_index[word])
                indexed_c.append(indexed_s)
            indexed_cs.append(np.array(indexed_c))
        return indexed_cs

    def _index_label(self, labels):
        indexed_ls = []
        for label in labels:
            indexed_ls.append(np.eye(self.c_max_len)[label])
        return indexed_ls

    def _index_question(self, questions):
        q_word_index = dict()
        for i, word in enumerate(self._q_word_set):
            q_word_index[word] = i+1  # index 0 for zero padding
        indexed_qs = []
        for sentence in questions:
            sentence = sentence.replace(".", " .")
            sentence = sentence.replace("?", " ?")
            sentence = sentence.split()
            indexed_s = []
            for word in sentence:
                indexed_s.append(q_word_index[word])
            indexed_qs.append(np.array(indexed_s))
        return indexed_qs

    def _index_answer(self, answers):
        a_word_index = dict()
        a_word_dict = dict()
        for i, word in enumerate(self._cqa_word_set):
            a_word_dict[i] = word
            if word in self._a_word_set:
                answer_one_hot = np.zeros(len(self._cqa_word_set), dtype=np.float32)
                answer_one_hot[i] = 1
                a_word_index[word] = answer_one_hot
        indexed_as = []
        for answer in answers:
            if ',' in answer:
                multiple_answer = [a_word_index[',']]
                for a in answer.split(','):
                    indexed_a = a_word_index[a]
                    multiple_answer.append(indexed_a)
                indexed_as.append(np.sum(multiple_answer, axis=0))
            else:
                indexed_a = a_word_index[answer]
                indexed_as.append(indexed_a)

        return indexed_as

    def masking(self, context_index, label_index, question_index):
        context_masked = []
        question_masked = []
        label_masked = []
        context_real_len = []
        question_real_len = []
        # cs: one context
        for cs, l, q in zip(context_index, label_index, question_index):
            context_masked_tmp = []
            context_real_length_tmp = []
            # cs: many sentences
            for context in cs:
                context_real_length_tmp.append(len(context))
                diff = self.s_max_len - len(context)
                if (diff > 0):
                    context_mask = np.append(context, [self.mask_index]*diff, axis=0)
                    context_masked_tmp.append(context_mask.tolist())
                else:
                    context_masked_tmp.append(context)
            diff_c = self.c_max_len - len(cs)
            context_masked_tmp.extend([[0]*self.s_max_len]*diff_c)
            context_masked.append(context_masked_tmp)

            diff_q = self.q_max_len - len(q)
            question_real_len.append(len(q))
            question_masked_tmp = np.array(np.append(q, [self.mask_index]*diff_q, axis=0))
            question_masked.append(question_masked_tmp.tolist())

            diff_l = self.c_max_len - len(l)
            label_masked_tmp = np.append(l, np.zeros((diff_l, self.c_max_len)), axis=0)
            label_masked.append(label_masked_tmp.tolist())
            context_real_length_tmp.extend([0] * diff_l)
            context_real_len.append(context_real_length_tmp)

        return context_masked, question_masked, label_masked, context_real_len, question_real_len

    def load(self, mode, path):

        assert mode in ['train', 'val', 'test']

        contexts, labels, questions, answers = self.split_all_clqa([path])

        context_index = self._index_context(contexts)
        label_index = self._index_label(labels)
        question_index = self._index_question(questions)
        answer_index = self._index_answer(answers)

        if mode == 'train':
            # check max sentence length
            for context in context_index:
                for sentence in context:
                    if len(sentence) > self.s_max_len:
                        self.s_max_len = len(sentence)
            # check max question length
            for question in question_index:
                if len(question) > self.q_max_len:
                    self.q_max_len = len(question)

            assert self.s_max_len > 0
            assert self.q_max_len > 0

            self.path_to_processed = '_'.join([
                self.output_path,
                str(self.c_max_len),
                str(self.s_max_len),
                str(self.q_max_len),
                str(len(self._c_word_set)),
                str(len(self._q_word_set)),
                str(len(self._a_word_set)),
            ])
            if not os.path.exists(self.path_to_processed):
                os.makedirs(self.path_to_processed)

        context_masked, question_masked, label_masked, context_real_len, question_real_len = self.masking(context_index, label_index, question_index)
        # check masking
        cnt = 0
        for c, q, l in zip(context_masked, question_masked, label_masked):
            for context in c:
                if (len(context) != self.s_max_len) | (len(q) != self.q_max_len) | (len(l) != self.c_max_len):
                    cnt += 1
        if cnt == 0:
            print("Masking success!")
        else:
            print("Masking process error")
        dataset = (question_masked, answer_index, context_masked, label_masked, context_real_len, question_real_len)

        dump_path = os.path.basename(path) + '.pkl'
        with open(os.path.join(self.path_to_processed, dump_path), 'wb') as f:
            pickle.dump(dataset, f)


def get_args_parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--path', required=True)
    _parser.add_argument('--c_max_len', type=int, required=True)
    _parser.add_argument('--all', '--all_paths', required=True)
    _parser.add_argument('--word_set', '--word_set_path', default=None,
                         help='Optional word set. If not specified, generated from'
                              'the union of training, validation, and test data.')
    _parser.add_argument('--output_path', required=True)

    return _parser


def default_write(f, string, default_value):
    if string is None:
        f.write(str(default_value) + "\t")
    else:
        f.write(str(string) + "\t")


def main():
    args = get_args_parser().parse_args()

    preprocess = Preprocess(args.path, args.c_max_len)

    preprocess.output_path = args.output_path
    preprocess.set_path(args.path, args.all)
    preprocess.set_word_set(args.word_set)

    for train_path in preprocess.train_paths:
        preprocess.load('train', train_path)
    for val_path in preprocess.val_paths:
        preprocess.load('val', val_path)
    for test_path in preprocess.test_paths:
        preprocess.load('test', test_path)


if __name__ == '__main__':
    main()
