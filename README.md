## Current Scenario!

There are thousands of millions of pages on the web, all ready to present the information on a variety of interesting and amusing topics. The Search Engines are the messengers of the same information at your disposal whenever you need them. Well, if you go by the technical definition as quoted by [Wikipedia](https://en.wikipedia.org/wiki/Web_search_engine):
“A web search engine is a software system that is designed to search for information on the World Wide Web. The search results are generally presented in a line of results often referred to as search engine results pages (SERPs)”

### The working

Every Search Engines use different complex mathematical algorithms for generating Search Results. Different Search Engines perceive different elements of a web page including page title, content, meta description and then come up with their results to rank on.
The 3 main functions of a Search Engine are:

1. Crawling: A crawler is a Search Engine bot or a Search Engine spider that travels all around the web looking out for new pages ready to be indexed.
2. Indexing: Once the Search Engines crawls the web and comes across the new pages, it then indexes or stores the information in its giant database categorically.
3. Providing information: Whenever a user types in his/her query and presses the enter button, the Search Engines would quest its directory of documents/information (that has already been crawled and indexed) and come back with the most relevant and popular results.

### Why Vison?

These search engines help with searching through words or phrases but have you ever used a search engine that searches with a picture or a short video clip?? Personally, we haven't come across anything that even sounds like it.

So The Code Foundation is going all out with its new project "The Vison" which gives you the one and only, one of its kind search engine Vison which enables quick search through images, audio and video.

1. Our especially designed crawler program will travel all over the web and download multimedia(images, videos etc) contents on our servers.

2. We will then index based on various techniques to collect, parse and store data to facilitate fast and accurate information retrival. 

3. As per the search query Vison would look into it's indexed data and as per the ranking of the content throw back the most relevant and popular results.

### Instructions (To get contributing started)

#### Requirements:

1. Anaconda
2. Tensorflow (>1.9.0) 

#### Steps:

1. Clone this repository in your local directory
2. Run __download_model.py__ script. This will download and extract the frozen tensorflow model. By default we are using 'SSD with mobilenet'. 
3. Next run __objectdetected_dict.py__ script. This will load the frozen model into the memory, load the path to images directory and run object detection on each image. This will output a dictionary containing objects detected in images along with there confidence point. 
