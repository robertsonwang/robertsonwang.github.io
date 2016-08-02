try:
	from setuptools import setup

except ImportError:
	from distutils.core import setup

config = {'description': 'My Project',
'author': 'Robertson Wang',
'url': 'http://robertsonwang.github.io/', 'download_url': 'Where to download it.', 'author_email': 'robertsonwang@gmail.com',
'version': '0.1', 'install_requires': ['nose'], 'packages': ['NAME'], 'scripts': [],
'name': 'projectname'}

setup(**config)
##** indicates arguments for dictionaries, the keys become separate keyword arguments
##This passes to setup, a builtin function, all of your contact information. That way, when people access
#this module in terminal and type --url, --download_url, etc they get information specific to YOUR project