#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import setuptools

setuptools.setup(name='myPortfolioManagement',
version='1.0',
description='This package develops utilities that allows customization in the portfolio management of retail Investors',
url='#',
author='Safis Hajjouz',
install_requires=['timebudget','quandl',
                  'tsmoothie', 'tslearn',
                  'pandas','numpy', 'datetime', 'quantstats',
                  'ray', 'investpy', 'finvizfinance',
                  'yahoofinancials','yfinance'],
dependency_links=['https://github.com/leopd/timebudget.git'],
author_email='shajjouz@gmail.com',
packages=setuptools.find_packages(),
zip_safe=False)

