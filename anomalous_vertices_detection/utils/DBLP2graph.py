# !/usr/bin/python
# wget -N http://dblp.uni-trier.de/xml/dblp.xml.gz
# then run this script
import gzip
import os
import csv
import DBLP2json


def force():
    print '** Computing coauthorship half-square graph...'

    allauthors = set()
    out = gzip.GzipFile('tmp_dblp_coauthorship.json.gz', 'w')
    csvwriter = csv.writer(out, delimiter=',')
    edgecount = 0
    for p, paper in enumerate(DBLP2json.papers()):
        tag, authors, year = paper
        for a, author1 in enumerate(authors):
            # allauthors.add(author1)
            for author2 in authors[a + 1:]:
                edgecount += 1
                csvwriter.writerow([author1.encode('utf8'), author2.encode('utf8')])
    out.close()
    os.rename('tmp_dblp_coauthorship.json.gz', 'dblp_coauthorship2.json.gz')

    print '--', len(allauthors), 'unique authors'
    print '--', edgecount, 'total coauthorship edges'


if __name__ == '__main__':
    force()
