import scrapy

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

from PIL import Image

#from scrapy.pipelines.images import ImagesPipeline
#https://doc.scrapy.org/en/latest/topics/media-pipeline.html

class ParliamentSpider(scrapy.Spider):
    name = "parliament"
    start_urls = [
        'https://www.aph.gov.au/Senators_and_Members/Members/Members_Photos'
    ]
    output = "images"

    def parse(self, response):
        # Get the list of images, and download them at the same time
        for img in response.css('ul.gallery li a.img-holder'):
            href = img.css('::attr(href)').extract_first()
            item = {
                #'id': href.split('=')[-1],
                'name': img.css('img::attr(alt)').extract_first(),
                'href': response.urljoin(href),
                'image': response.urljoin(img.css('img::attr(src)').extract_first())
            }
            self.log('url %s' % item['image'])
            yield scrapy.Request(url=item['image'], callback=self.parse_image, meta={'item':item})

    def parse_image(self, response):
        # Load the image and save callback
        item = response.meta['item']
        image = Image.open(BytesIO(response.body))
        filename = '{}/{}.jpeg'.format(self.output, item['name'])
        image.save(filename)
        self.log('download: {} {}'.format(image.size, filename))
        yield item
