import subprocess
import web

urls=('/', 'Index', '/cv', 'cv', '/projects', 'projects', '/research', 'research')

app = web.application(urls, globals())

render = web.template.render('templates/', base="layout")

class Index(object):
	def GET(self):
 		return render.index()
 		 		
class cv(object):
	def GET(self):
		pdf='/Users/robertsonwang/Desktop/Python/robertsonwang/bin/RobertsonWang-CV.pdf'
		subprocess.Popen(['open', pdf])
 		return render.cv()
 		
class projects(object):
	def GET(self):
 		return render.projects()

class research(object):
	def GET(self):
 		return render.research()

if __name__ == "__main__":
	app.run()