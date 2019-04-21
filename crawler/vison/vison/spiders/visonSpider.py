# -*- coding: utf-8 -*-
import scrapy
from scrapy.spiders import CrawlSpider
import sqlite3

class VisonspiderSpider(scrapy.Spider):
    name = 'visonSpider'
    start_urls = ['https://www.wikipedia.org/']

    def parse(self, response):
        # Cursor for the db
        conn = sqlite3.connect('vison.db')
        c = conn.cursor()
        # Set default meta information for first page
        from_url = ''
        from_text = ''
        depth = 0
        # Extract the meta information from the response, if any
        if 'from' in response.meta:
            from_url = response.meta['from']
        if 'text' in response.meta:
            from_text = response.meta['text']
        if 'depth' in response.meta:
            depth = response.meta['depth']

        # Update the print logic to show what page contain a link to the
        # current page, and what was the text of the link
        self.start_urls.append(response.url)
        print(depth, response.url, '<-', from_url, from_text, sep=' ')
        c.execute("INSERT INTO urls VALUES (?,?);", (response.url,1))
        conn.commit()
        conn.close()
        page = response.url.split("/")[-2]
        filename = '%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)

        a_selectors = response.xpath("//a")
        for selector in a_selectors:
            text = selector.xpath("text()").extract_first()
            link = selector.xpath("@href").extract_first()
            request = response.follow(link, callback=self.parse)
            # Meta information: URL of the current page
            request.meta['from'] = response.url
            # Meta information: text of the link
            request.meta['text'] = text
            # Meta information: depth of the link
            request.meta['depth'] = depth + 1
            yield request
