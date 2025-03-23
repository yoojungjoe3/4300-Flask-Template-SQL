LOAD DATA LOCAL INFILE '/workspaces/4300-Flask-Template-SQL/DataSet.csv'
INTO TABLE quillquestdb
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(Name, Fandom, Ships, Rating, Link, Review, Abstract);
