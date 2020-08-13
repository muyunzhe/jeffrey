#encoding: utf-8
import sys
import os
import random
from collections import Counter

reload(sys)
sys.setdefaultencoding('utf-8')
 
class MetaPathGenerator:
    def __init__(self):
        self.author_doc = dict()
        self.keyword_doc_ng1 = dict()
        self.keyword_doc_ng2 = dict()
        self.keyword_doc_ng3 = dict()
        self.doc_author = dict()
        self.doc_keyword_ng1 = dict()
        self.doc_keyword_ng2 = dict()
        self.doc_keyword_ng3 = dict()
        self.id_keyword = dict()
        self.total_docs = dict()
        self.doc_keyword = dict()
        self.keyword_doc = dict()

    def read_data(self, dirpath):
        i = 0
        with open(dirpath, 'r') as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) != 2:
                    continue
                [keyword, docs] = toks
                self.id_keyword[keyword] = str(i)
                i += 1
                docs = docs.split(' ')
                #for doc in docs:
                #    self.total_docs[doc] = 1
                #ng = len(keyword.split('\001'))
                #if ng == 1:
                #    self.keyword_doc_ng1[keyword] = docs
                #    for doc in docs:
                #        if doc not in self.doc_keyword_ng1:
                #            self.doc_keyword_ng1[doc] = []
                #        self.doc_keyword_ng1[doc].append(keyword)
                #if ng == 2:
                #    self.keyword_doc_ng2[keyword] = docs
                #    for doc in docs:
                #        if doc not in self.doc_keyword_ng2:
                #            self.doc_keyword_ng2[doc] = []
                #        self.doc_keyword_ng2[doc].append(keyword)
                #if ng == 3:
                #    self.keyword_doc_ng3[keyword] = docs
                #    for doc in docs:
                #        if doc not in self.doc_keyword_ng3:
                #            self.doc_keyword_ng3[doc] = []
                #        self.doc_keyword_ng3[doc].append(keyword)
                self.keyword_doc[keyword] = docs
 
                for doc in docs:
                    self.total_docs[doc] = 1
                    if doc not in self.doc_keyword:
                        self.doc_keyword[doc] = []
                    self.doc_keyword[doc].append(keyword)
        print "load keyword2doclist success, keywordlength " + str(len(self.id_keyword))
        fkwordid = open(dirpath + "keyword2id_pub_lexer", 'wb')
        for keyword, index in self.id_keyword.items():
            fkwordid.write(keyword + '\t' + index + '\n')
        #print "#authors", len(self.id_author)
        fkwordid.close()
        
    def generate_random_aca(self, outfilename, walklength, numwalks=None):
        outfile = open(outfilename, 'w')
        doc_list = list()
        for doc in self.total_docs:
            doc0 = doc
            #print "--------------------new_round!-------------------------"
            outline = self.generate_slave(doc0, self.keyword_doc, self.doc_keyword, walklength)
            #print '1 gram:'
            #outline_1 = self.generate_slave(doc0, self.keyword_doc_ng1, self.doc_keyword_ng1, walklength)
            #print '2 gram'
            #outline_2 = self.generate_slave(doc0, self.keyword_doc_ng2, self.doc_keyword_ng2, walklength)
            #print '3 gram'
            #outline_3 = self.generate_slave(doc0, self.keyword_doc_ng3, self.doc_keyword_ng3, walklength)
            outfile.write(outline)
        outfile.close()
    
    def generate_slave(self, doc0, keyword_doc, doc_keyword, walklength):
        total_outline = ''
        if doc0 not in doc_keyword:
            #print 'no doc in dict!'
            return total_outline
        keywords_num = len(doc_keyword[doc0])
        numwalks = keywords_num * 2
        #print 'keywords_num:%d, numwalks:%d' % (keywords_num, numwalks)
        for j in xrange(0, numwalks): #wnum walks
            try:
                doc = doc0
                outline = 'i' + doc0
                for i in xrange(0, walklength):
                    #doc2keyword
                    keywords = doc_keyword[doc]
                    numa = len(keywords)
                    keywordsid = random.randrange(numa)
                    keyword = keywords[keywordsid]
                    #if i == 0:
                    #    print keyword
                        
                    keyword_id = self.id_keyword.get(keyword)
                    outline += " " + 'vk' + str(keyword_id)
                    #keyword2doc
                    docs = keyword_doc[keyword]
                    numa = len(docs)
                    docid = random.randrange(numa)
                    doc = docs[docid]
                    outline += " " + 'i' + doc

                #outfile.write(outline + "\n")
                total_outline = total_outline + outline + "\n"
            except Exception as e:
                continue
        return total_outline
        
        
#python py4genMetaPaths.py 1000 100 net_aminer output.aminer.w1000.l100.txt
#python py4genMetaPaths.py 1000 100 net_dbis   output.dbis.w1000.l100.txt
 
#dirpath = "net_aminer"
#OR 
#dirpath = "net_dbis"
#OR
#dirpath = "./"
 
walklength = int(sys.argv[2])
 
dirpath = sys.argv[1]
outfilename = sys.argv[3]
 
def main():
    mpg = MetaPathGenerator()
    mpg.read_data(dirpath)
    mpg.generate_random_aca(outfilename, walklength)
 
 
if __name__ == "__main__":
    main()
