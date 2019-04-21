## Crawler

- Downloads
1. Sqlite - [https://www.sqlite.org/download.html]here (make sure to add it to your path)

- Packages needed
1. Scrapy - [https://docs.scrapy.org/en/latest/intro/install.html]here

After succesfull installation. Move to the vison directory.
Create a sqlite database as follows:
```bash
    $ cd vison
    $ sqlite3
```
After this you will be able to execute sqlite3 commands from the terminal
```bash
    > .open vison.db
    > create tables urls (
    > urlId text,
    > rank integer);
```
This command will set up the new table to save the urls.
Exit the sqlite prompt and execute the following command:

```bash
    $ scrapy crawl visonSpider
```
