LOAD DATA INFILE '/var/lib/mysql-files/DataSet_clean.csv'
INTO TABLE fanfics
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Name, Fandom, Ships, Rating, Link, Review, Abstract);
