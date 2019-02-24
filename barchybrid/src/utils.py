from collections import Counter
import re
import os, time
from itertools import chain
from operator import itemgetter
import random
import codecs


class ConllEntry:
    def __init__(self, id, form, lemma, pos, cpos, feats=None, parent_id=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.language_id = None
        self.language_name = None

        self.pred_pos = None
        self.pred_cpos = None


    def clone(self):
        return ConllEntry(
            self.id, self.form, self.lemma, self.pos, self.cpos,
            self.feats, self.parent_id, self.relation, self.deps,
            self.misc
        )

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, \
                  self.pred_pos if self.pred_pos else self.pos,\
                  self.pred_cpos if self.pred_cpos else self.cpos,\
                  self.feats, str(self.pred_parent_id) if self.pred_parent_id \
                  is not None else str(self.parent_id), self.pred_relation if\
                  self.pred_relation is not None else self.relation, \
                  self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


class Treebank(object):
    def __init__(self, trainfile, devfile, testfile):
        self.name = 'noname'
        self.trainfile = trainfile
        self.devfile = devfile
        self.dev_gold = devfile
        self.test_gold = testfile
        self.testfile = testfile
        self.outfilename = None


class UDtreebank(Treebank):
    def __init__(self, treebank_info, location, shared_task=False, shared_task_data_dir=None):
        """
        Read treebank info to a treebank object
        The treebank_info element contains different information if in shared_task mode or not
        If not: it contains a tuple with name + iso ID
        Else: it contains a dictionary with some information
        """
        if shared_task:
            self.lcode = treebank_info['lcode']
            if treebank_info['tcode'] == '0':
                self.iso_id = treebank_info['lcode'] # iso id becomes the language code
            else:
                self.iso_id = treebank_info['lcode'] + '_' + treebank_info['tcode']
            #self.testfile = location + self.iso_id + '.conllu'
            self.testfile = location + self.iso_id + '.txt' # if using UUsegmenter as input
            if not os.path.exists(self.testfile):
                self.testfile = shared_task_data_dir + self.iso_id + '.conllu'
            self.dev_gold = shared_task_data_dir + self.iso_id + '.conllu'
            self.test_gold = shared_task_data_dir + self.iso_id + '.conllu'
            self.outfilename = treebank_info['outfile']
        else:
            self.name, self.iso_id = treebank_info
            files_prefix = location + "/" + self.name + "/" + self.iso_id
            self.trainfile = files_prefix + "-ud-train.conllu"
            self.devfile = files_prefix + "-ud-dev.conllu"
            self.testfile = files_prefix + "-ud-dev.conllu"
            self.test_gold= files_prefix + "-ud-dev.conllu"
            self.dev_gold= files_prefix + "-ud-dev.conllu"
            self.outfilename = self.iso_id + '.conllu'


class ParseForest:
    def __init__(self, sentence):
        self.roots = list(sentence)

        for root in self.roots:
            root.children = []
            root.scores = None
            root.parent = None
            root.pred_parent_id = None
            root.pred_relation = None
            root.vecs = None
            root.lstms = None

    def __len__(self):
        return len(self.roots)


    def Attach(self, parent_index, child_index):
        parent = self.roots[parent_index]
        child = self.roots[child_index]

        child.pred_parent_id = parent.id
        del self.roots[child_index]


def isProj(sentence):
    forest = ParseForest(sentence)
    unassigned = {entry.id: sum([1 for pentry in sentence if pentry.parent_id == entry.id]) for entry in sentence}

    for _ in xrange(len(sentence)):
        for i in xrange(len(forest.roots) - 1):
            if forest.roots[i].parent_id == forest.roots[i+1].id and unassigned[forest.roots[i].id] == 0:
                unassigned[forest.roots[i+1].id]-=1
                forest.Attach(i+1, i)
                break
            if forest.roots[i+1].parent_id == forest.roots[i].id and unassigned[forest.roots[i+1].id] == 0:
                unassigned[forest.roots[i].id]-=1
                forest.Attach(i, i+1)
                break

    return len(forest.roots) == 1


def vocab(conll_path, path_is_dir=False):
    """
    Collect frequencies of words, cpos, pos and deprels + languages.
    """
    wordsCount = Counter()
    charsCount = Counter()
    posCount = Counter()
    cposCount = Counter()
    relCount = Counter()
    langCounter = Counter()

    if path_is_dir:
        data = read_conll_dir(conll_path, "train")
    else:
        data = read_conll(conll_path, vocab_prep=True)

    for sentence in data:
        wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
        for node in sentence:
            if isinstance(node, ConllEntry) and not node.form == u"*root*":
                charsCount.update(node.form)

        posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
        cposCount.update([node.cpos for node in sentence if isinstance(node, ConllEntry)])
        relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

        if path_is_dir:
            langCounter.update([node.language_id for node in sentence if
                                isinstance(node, ConllEntry)])
    print "=== Vocab Statistics: ==="
    print "Vocab containing %d words" % len(wordsCount)
    print "Charset containing %d chars" % len(charsCount)
    print "UPOS containing %d tags" % len(posCount)
    print "CPOS containing %d tags" % len(cposCount)
    print "Rels containing %d tags" % len(relCount)
    print "Langs containing %d langs" % len(langCounter)

    return (wordsCount, {w: i for i, w in enumerate(wordsCount.keys())},
            posCount.keys(), cposCount.keys(), relCount.keys(),
            langCounter.keys() if langCounter else None, charsCount.keys())


def conll_dir_to_list(
    languages, data_dir,
    shared_task=False,
    shared_task_data_dir=None,
    treebanks_from_json=True,
):
    import json
    if not treebanks_from_json:
        print "Scanning for available treebanks in", data_dir
        treebank_metadata = []
        for entry in os.listdir(data_dir):
            candidate_dir = os.path.join(data_dir, entry)
            if os.path.isdir(candidate_dir):
                for filename in os.listdir(candidate_dir):
                    fields = filename.split('-ud-')
                    if len(fields) == 2 and fields[1] == 'train.conllu':
                        treebank_metadata.append((
                            entry.decode('utf-8'), # tbname
                            fields[0].decode('utf-8') # tbid
                        ))

    elif shared_task:
        print "Reading available treebanks from shared task metadata"
        metadataFile = shared_task_data_dir +'/metadata.json'
        metadata = codecs.open(metadataFile, 'r', encoding='utf-8')
        json_str = metadata.read()
        treebank_metadata = json.loads(json_str)
    else:
        print "Reading available treebanks from hard-coded json file"
        ud_iso_file = codecs.open('./src/utils/ud_iso.json', encoding='utf-8')
        json_str = ud_iso_file.read()
        iso_dict = json.loads(json_str)
        treebank_metadata = iso_dict.items()
    ud_treebanks = [UDtreebank(treebank_info, data_dir, shared_task, shared_task_data_dir) \
            for treebank_info in treebank_metadata ]
    return ud_treebanks


def read_conll_dir(languages, filetype, maxSize=-1):
    #print "Max size for each corpus: ", maxSize
    if filetype == "train":
        return chain(*(read_conll(lang.trainfile, lang.name, maxSize) for lang in languages))
    elif filetype == "dev":
        return chain(*(read_conll(lang.devfile, lang.name) for lang in languages if lang.pred_dev))
    elif filetype == "test":
        return chain(*(read_conll(lang.testfile, lang.name) for lang in languages))


def read_conll(filename, language=None, maxSize=-1, hard_lim=False, vocab_prep=False, drop_nproj=False):
    # hard lim means capping the corpus size across the whole training procedure
    # soft lim means using a sample of the whole corpus at each epoch
    fh = codecs.open(filename, 'r', encoding='utf-8')
    print "Reading " + filename
    has_tbweights = False
    if filename.endswith('.conllu'):
        tbweights_filename = filename[:-7] + '.tbweights'
        # Ideally, we should test whether om.tbweights_from_file is set
        # but it would require lots of changes to get it here.
        if os.path.exists(tbweights_filename):
            tbweights_f = open(tbweights_filename, 'rb')
            has_tbweights = True
            print "Also reading " + tbweights_filename

    if vocab_prep and not hard_lim:
        maxSize = -1 # when preparing the vocab with a soft limit we need to use the whole corpus
    ts = time.time()
    dropped = 0
    unable_to_find_parent = 0
    read = 0
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-CPOS', '_', -1, 'rroot', '_', '_')
    root.language_id = language
    tokens = [root]
    tb_index = 0
    yield_count = 0
    if maxSize > 0 and not hard_lim:
        all_tokens = []
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            # empty line --> complete sentence in list `tokens`
            if len(tokens) > 1:
                new_root = root.clone()
                new_root.language_id = root.language_id
                new_root.tb_index = tb_index
                if has_tbweights:
                    new_root.tb_weights = tbweights_f.readline().rstrip()
                    if not new_root.tb_weights:
                        raise ValueError, 'not enough lines in %r' %tbweights_filename
                conll_tokens = [t for t in tokens if isinstance(t, ConllEntry)]
                if not drop_nproj or isProj(conll_tokens): # keep going if it's projective or we're not dropping non-projective sents
                #dropping the proj for exploring swap
                #if not isProj([t for t in tokens if isinstance(t, ConllEntry)]):
                    inorder_tokens = inorder(conll_tokens)
                    for i, t in enumerate(inorder_tokens):
                        t.projective_order = i
                    for tok in conll_tokens:
                        tok.rdeps = [i.id for i in conll_tokens if i.parent_id == tok.id]
                        if tok.id != 0:
                            try:
                                tok.parent_entry = [i for i in conll_tokens if i.id == tok.parent_id][0]
                            except IndexError:
                                unable_to_find_parent += 1
                    if maxSize > 0:
                        if not hard_lim:
                            all_tokens.append(tokens)
                        else:
                            yield tokens
                            yield_count += 1
                            if yield_count == maxSize:
                                print "Capping size of corpus at " + str(yield_count) + " sentences"
                                break;
                    else:
                        yield tokens
                else:
                    # drop_nproj and not isProj()
                    dropped += 1
                read += 1
            else:
                # dropping sentence with no tokens and comments, i.e. two empty lines
                print 'Warning: Found sentence with no tokens and comments, i.e. two empty lines. File not in .conllu format.'
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                token = ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9])
                token.language_id = language
                tokens.append(token)
    if len(tokens) > 1:
        new_root = root.clone()
        new_root.language_id = root.language_id
        new_root.tb_index = tb_index
        if has_tbweights:
            new_root.tb_weights = tbweights_f.readline().rstrip()
            if not new_root.tb_weights:
                raise ValueError, 'not enough lines in %r' %tbweights_filename
        tokens[0] = new_root
        yield tokens
    if has_tbweights:
        line = tbweights_f.readline()
        if line:
            raise ValueError, 'too many lines in %r' %tbweights_filename
        tbweights_f.close()
    if dropped:
        print 'Warning: dropped %d sentence(s)' %dropped
    if unable_to_find_parent:
        print 'Warning (not relevant in predict mode): was not able to find parent %d time(s)' %unable_to_find_parent
    if hard_lim and yield_count < maxSize:
        print 'Warning: unable to yield ' + str(maxSize) + ' sentences, only ' + str(yield_count) + ' found'
    if len(tokens) > 1:
        print 'Warning: found tokens that are not followed by a sentence-ending empty line; ignoring trailing data'

    print read, 'sentences read'

    if maxSize > 0 and not hard_lim:
        random.shuffle(all_tokens)
        all_tokens = all_tokens[:maxSize]
        print "Yielding " + str(len(all_tokens)) + " random sentences"
        for toks in all_tokens:
            yield toks

    te = time.time()
    print 'Time: %.2gs'%(te-ts)


def write_conll(fn, conll_gen):
    print "Writing to " + fn
    sents = 0
    with codecs.open(fn, 'w', encoding='utf-8') as fh:
        for sentence in conll_gen:
            sents += 1
            for entry in sentence[1:]:
                fh.write(unicode(entry) + '\n')
                #print str(entry)
            fh.write('\n')
        print "Wrote " + str(sents) + " sentences"


def write_conll_multiling(conll_gen, languages):
    lang_dict = {language.name:language for language in languages}
    cur_lang = conll_gen[0][0].language_id
    outfile = lang_dict[cur_lang].outfilename
    fh = codecs.open(outfile, 'w', encoding='utf-8')
    print "Writing to " + outfile
    for sentence in conll_gen:
        if cur_lang != sentence[0].language_id:
            fh.close()
            cur_lang = sentence[0].language_id
            outfile = lang_dict[cur_lang].outfilename
            fh = codecs.open(outfile, 'w', encoding='utf-8')
            print "Writing to " + outfile
        for entry in sentence[1:]:
            fh.write(unicode(entry) + '\n')
        fh.write('\n')


def parse_list_arg(l):
    """Return a list of line values if it's a file or a list of values if it
    is a string"""
    if os.path.isfile(l):
        f = codecs.open(l, 'r', encoding='utf-8')
        return [line.split()[0] for line in f]
    else:
        # also support colon as a separator
        l = l.replace(':', ' ')
        return l.split()


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");


def normalize(word):
    return 'NUM' if numberRegex.match(word) else word.lower()


def strong_normalize(word):
    if numberRegex.match(word):
        return 'NUM'
    else:
        word = word.lower()
        word = re.sub(r".+@.+", "*EMAIL*", word)
        word = re.sub(r"@\w+", "*AT*", word)
        word = re.sub(r"(https?://|www\.).*", "*url*", word)
        word = re.sub(r"([^\d])\1{2,}", r"\1\1", word)
        word = re.sub(r"([^\d][^\d])\1{2,}", r"\1\1", word)
        word = re.sub(r"``", '"', word)
        word = re.sub(r"''", '"', word)
        word = re.sub(r"\d", "0", word)
        return word


def evaluate(gold, test, conllu):
    scoresfile = test + '.txt'
    print "Writing to " + scoresfile
    if not conllu:
        os.system('perl src/utils/eval.pl -g ' + gold + ' -s ' + test  + ' > ' + scoresfile )
    else:
        os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + gold + ' ' + test + ' > ' + scoresfile) # need to switch to conll18
    score = get_LAS_score(scoresfile, conllu)
    return score


def set_seeds(options):
    # note that dynet sets it seed at module loading time
    if options.dynet_seed:
        if  options.use_default_seed:
            print 'Using 1 as the Python seed (and we see that a DyNet seed was specified)'
            random.seed(1)
        else:
            print 'Setting Python seed to match DyNet seed'
            random.seed(options.dynet_seed)
    else:
        if  options.use_default_seed:
            print 'Using 1 as the Python seed (and there is no information on a DyNet seed)'
            random.seed(1)
        else:
            print 'Not setting Python seed (and there is no information on a DyNet seed)'


def inorder(sentence):
    queue = [sentence[0]]
    def inorder_helper(sentence, i):
        results = []
        left_children = [entry for entry in sentence[:i] if entry.parent_id == i]
        for child in left_children:
            results += inorder_helper(sentence, child.id)
        results.append(sentence[i])

        right_children = [entry for entry in sentence[i:] if entry.parent_id == i ]
        for child in right_children:
            results += inorder_helper(sentence, child.id)
        return results
    return inorder_helper(sentence, queue[0].id)


def set_python_seed(seed):
    random.seed(seed)


def generate_seed():
    return random.randint(0,10**9) # this range seems to work for Dynet and Python's random function


def get_LAS_score(filename, conllu=True):
    score = None
    with codecs.open(filename,'r',encoding='utf-8') as fh:
        if conllu:
            for line in fh:
                if re.match(r'^LAS',line):
                    elements = line.split()
                    score = float(elements[6]) # should extract the F1 score
        else:
            las_line = [line for line in fh][0]
            score = float(las_line.split('=')[1].split()[0])
    return score
