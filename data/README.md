## Data
| Network               | Directed | Labeled   | Vertices  | Edges      |
|-----------------------|----------|-----------|-----------|------------|
| [Academia][1]         | True     | False     | 200,169   | 1,389,063  |
| [ArXiv HepPh][2]      | Fasle    | False     | 34,546    | 421,578    |
| [CLASS OF 1880/81][3] | True     | [True][8] | 53        | 179        |
| [DBLP][4]             | False    | False     | 1,655,850 | 13,504,952 |
| [Flixster][5]         | True     | False     | 672,827   | 1,099,003  |
| [Twitter][6]          | True     | [True][9] | 5,384,160 | 16,011,443 |
| [Yelp][7]             | Fasle    | Fasle     | 249,443   | 3,563,818  |

### Academia
[Academia.edu][10] is a social platform for academics to share and follow research, and to follow other researchers’ work. Using our dedicated crawler,
we crawled most of the Academia.edu graph during 2011.
###ArXiv HepPh
[ArXiv][11] is an e-print service in fields such as physics and computer science.
We used the ArXiv HEP-PH (high energy physics phenomenology) citation graph that was released as part of the 2003 KDD Cup. [[Source][17]]
### CLASS OF 1880/81
[CLASS OF 1880/81][12] is a dataset which contains the friendship network of a German school class from 1880-81 that was assembled by the class’s primary school teacher, Johannes Delitsch.
The dataset itself was generated from observing students, interviewing pupils and parents, and analyzing school essays .
Delitsch found that there were 13 outliers out of 53 students, which Heidler et al. defined as students who did not fit perfectly into their predicted position within the network structure.
The data contains three types of outliers: “repeaters,” who were four
students who often led the games; “sweets giver,” a student who bought
his peers’ friendship with candies; and a specific group of seven students
who were psychologically or physically handicapped, or socio-economically deprived.
This is probably the first-ever primarily collected social network dataset. [[Source][18]]
### DBLP
[DBLP][13] is  is the online reference for bibliographic information on major computer science publications.
We used a version of the DBLP dataset to build a co-authorship graph where two authors are connected if they published
a publication together. [[Source][19]]
### Flixster
[Flixster][14] is a social movie site which allows users to share movie reviews, discover new movies, and communicate with others.
We collected the data using a dedicated crawler during 2012
### Twitter
[Twitter][15] is an undirected online social network where people publish
short messages and updates. Currently, Twitter has 310 million monthly
active users. According to recent reports, Twitter has a bot infestation
problem. We used a dedicated API crawler to obtain our dataset
in 2014.
### Yelp
[Yelp][16] is is a web platform to help people find local businesses.
In addition to finding local business and writing reviews, Yelp allows its users to discover
events, make lists, and talk with other Yelpers.
In 2016 Yelp published several big datasets as part of the Yelp Dataset Challenge; one is a social network of its users. [[Source][20]]


[1]: http://proj.ise.bgu.ac.il/sns/datasets/academia.csv.gz
[2]: http://proj.ise.bgu.ac.il/sns/datasets/Cit-HepPh.txt.csv.gz
[3]: http://proj.ise.bgu.ac.il/sns/datasets/Relationship_patterns_in_the_19th_century.csv
[4]: http://proj.ise.bgu.ac.il/sns/datasets/dblp_coauthorship.csv.csv.gz
[5]: http://proj.ise.bgu.ac.il/sns/datasets/flixster.csv.gz
[6]: http://proj.ise.bgu.ac.il/sns/datasets/twitter.csv.gz
[7]: http://proj.ise.bgu.ac.il/sns/datasets/yelp_user_graph.csv.csv.gz
[8]: http://proj.ise.bgu.ac.il/sns/datasets/Relationship_patterns_labels.csv
[9]: http://proj.ise.bgu.ac.il/sns/datasets/twitter_fake_ids.csv
[10]: https://www.academia.edu/
[11]: https://arxiv.org
[12]: http://www.sciencedirect.com/science/article/pii/S0378873313000865
[13]: http://dblp.uni-trier.de/
[14]: http://flixster.com/
[15]: http://twitter.com
[16]: http://yelp.com
[17]: https://snap.stanford.edu/data/cit-HepPh.html
[18]: https://github.com/gephi/gephi/wiki/Datasets
[19]: http://dblp.uni-trier.de/xml/
[20]: https://www.yelp.com/dataset_challenge