from bs4 import BeautifulSoup
import urllib.request
import os
from selenium import webdriver
import pdfkit

class HTML_parser(object):

	def __init__(self, tickers):
		self.tickers = tickers
	
	def find_links(self, tickers):
		test = open('test.txt', 'w+')
		dataclean = []
		
		#for a in range(len(tickers)):
		data = []
		count=0
		
		try:
			urllib.request.urlretrieve("http://secfilings.nasdaq.com/filingsCompany.asp?SchParam=Ticker&SchValue="+str(tickers)+"&StartRow=1&EndRow=10000", 'test.txt')
		except:
			print(tickers + " is not a valid stock ticker")
			
		soup = BeautifulSoup(test, 'html.parser')
		table = soup.find_all(target="fetchFiling")
		
		for a in table:
			if '10-Q' in a['href']:
				data.append(a['href'])
		
		while count < len(data):
			try:
 				dataclean.append("http://secfilings.nasdaq.com"+str(data[count]))
 				count = count+3 #This is IDIOSYNCRATIC to the nasdaq source, duplicates may not always come in pairs of 3
			except:
				None
		return dataclean
 					
class pdf_writer(object):

	def __init__(self, tick_list):
		#self.cleandata = cleandata
		self.tick_list = tick_list
		
	def write_pdf(self, tick_list):
		dir=os.getcwd()
		
		for b in range(len(tick_list)):
			cleandata = HTML_parser(tick_list[b]).find_links(tick_list[b])
			driver = webdriver.PhantomJS()
			
			for a in range(len(cleandata)):
			#Specify the name of the pdf file:
				begin_date_index, end_date_index = str(cleandata[a]).find("RcvdDate=")+len("RcvdDate="), str(cleandata[a]).find("&CoName")
				date = str(cleandata[a])[begin_date_index:end_date_index]
				date = date.replace("/","_")
				temp = dir+'/'+str(tick_list[b])+"_"+date+"_10_Q.pdf"
				
				#Write the PDF
				driver.get(cleandata[a])
				driver.switch_to_frame(1)
				pdfkit.from_url(driver.current_url, temp)

#Tickers is the user input, which we pass to the HTML_Parser class to clean up the links from the HTML file

input_tickers = input("For which tickers do you want the earnings reports for? Separate with a comma \n")
tickers = [x.strip() for x in input_tickers.split(',')]
pdf_writer(tickers).write_pdf(tickers)
 