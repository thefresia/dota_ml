import scrapy
from scrapy_splash import SplashRequest
from bs4 import BeautifulSoup


class QuotesSpider(scrapy.Spider):
    name = "od_matches"
    opendotaUrl = 'https://www.opendota.com/'
    dotabuffUrl = 'https://ru.dotabuff.com/'
    user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:67.0) Gecko/20100101 Firefox/67.0'
    id = 0
    failed = 0

    def start_requests(self):
        url = QuotesSpider.dotabuffUrl + 'matches/'
        id = 4887481539
        QuotesSpider.id = id
        amount = 1000

        for i in range(id, id-amount,-1):
            matchUrl = url + str(i)
            yield SplashRequest(
                url=matchUrl, 
                callback=self.parse,         
                headers={
                        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:67.0) Gecko/20100101 Firefox/67.0',
                        },
                endpoint='render.html'
            )

    def parse(self, response):
        filename = 'data/match-%s.html' % QuotesSpider.id
        if (response.status == 404):
            QuotesSpider.failed = QuotesSpider.failed + 1
        print('======================')
        print('======================')
        print(QuotesSpider.failed)
        print('======================')
        print('======================')
        