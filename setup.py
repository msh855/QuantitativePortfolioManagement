#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import setuptools

setuptools.setup(name='myPortfolioManagement',
version='0.1',
description='This package develops utilities that allows customization in the portfolio management of retail Investors',
url='#',
author='Safis Hajjouz',
install_requires=['timebudget', 'itertools','quandl',
                  'tsmoothie', 'tslearn',
                  'pandas','numpy', 'datetime',
                  'ray', 'investpy', 'finvizfinance',
                  'yahoofinancials','yfinance'],
dependency_links=['https://github.com/leopd/timebudget.git'],
author_email='shajjouz@gmail.com',
packages=setuptools.find_packages(),
zip_safe=False)

