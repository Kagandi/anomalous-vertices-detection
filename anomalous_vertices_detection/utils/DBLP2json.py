# Source: projects.csail.mit.edu/dnd/DBLP/DBLP2json.py
#!/usr/bin/python
# wget -N http://dblp.uni-trier.de/xml/dblp.xml.gz
# then run this script
import gzip
import json
import os
import sys
import xml.sax

xml_filename = 'dblp.xml.gz'
json_gz_filename = 'dblp.json.gz'
tmp_filename = 'tmp_dblp.json.gz'
report_frequency = 10000


class DBLPHandler(xml.sax.ContentHandler):
    papertypes = {'article', 'book', 'inproceedings', 'incollection', 'www', 'proceedings', 'phdthesis',
                  'mastersthesis'}

    def __init__(self, out):
        self.out = out
        self.paper = None
        self.authors = []
        self.year = None
        self.text = ''
        self.papercount = 0
        self.edgecount = 0

    def startElement(self, name, attrs):
        if name in self.papertypes:
            self.paper = str(attrs['key'])
            self.authors = []
            self.year = None
        elif name in ['author', 'year']:
            self.text = ''

    def endElement(self, name):
        if name == 'author':
            self.authors.append(self.text)
        if name == 'year':
            self.year = int(self.text.strip())
        elif name in self.papertypes:
            self.write_paper()
            self.paper = None

    def write_paper(self):
        if self.papercount:
            self.out.write(',\n')
        self.papercount += 1
        self.edgecount += len(self.authors)
        json.dump([self.paper, self.authors, self.year], self.out)
        if self.papercount % report_frequency == 0:
            print '... processed %d papers, %d edges so far ...' % (self.papercount, self.edgecount)
            sys.stdout.flush()

    def characters(self, chars):
        self.text += chars


def force():
    print '** Parsing XML...'

    xmlfile = gzip.GzipFile(xml_filename, 'r')
    out = gzip.GzipFile(tmp_filename, 'w')
    out.write('[\n')
    dblp = DBLPHandler(out)
    parser = xml.sax.parse(xmlfile, dblp)
    out.write('\n]\n')
    out.close()
    os.rename(tmp_filename, json_gz_filename)

    print '-- %d papers, %d edges' % (dblp.papercount, dblp.edgecount)


def main(parse_args=False):
    try:
        need = (os.stat(xml_filename).st_mtime >= os.stat(json_gz_filename).st_mtime)
    except OSError:
        need = True
    if parse_args and len(sys.argv) > 1:
        need = True
    if need:
        force()


def open_gzip():
    main()
    return gzip.GzipFile(json_gz_filename, 'r')


def papers():
    for line in open_gzip():
        if line.strip() in '[]':
            continue
        line = line.rstrip().rstrip(',')
        yield json.loads(line)


if __name__ == '__main__':
    main(True)
