#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import setuptools

setuptools.setup(name='myPortfolioManagement',
                 version='1.0',
                 description='A package dedicated for Quantitative Portfolio Management for Retail-Investors',
                 url='#',
                 author='Safis Hajjouz',
                 install_requires=['timebudget', 'quandl',
                                   'tsmoothie', 'tslearn', 'maData',
                                   'pandas', 'numpy', 'datetime', 'quantstats',
                                   'ray', 'investpy', 'finvizfinance',
                                   'yahoofinancials', 'yfinance'],
                 dependency_links=['https://github.com/leopd/timebudget.git'],
                 author_email='shajjouz@gmail.com',
                 packages=setuptools.find_packages(),
                 zip_safe=False)
