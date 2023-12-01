from openbb_terminal.sdk import openbb

# put my keys
openbb.keys.quandl(key="fhbmNKX6oNP7PpFuZJNo")
openbb.keys.fred(key="cc628b51e21828ae6b98c06f4eef6714")
openbb.keys.fmp(key="eb50221eaef20292fe4b57f675be8b23")
openbb.keys.tradeconomics(key = '3e69f3322ec04da:riqejdprpxb0e5c')
from openbb import obb
obb.account.login(pat="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdXRoX3Rva2VuIjoiUkNOdzV6V3hSalNGRkV6Zjh5bXQ3NHpmSG9RZ2hQQTlkc2xvQ0hDUyIsImV4cCI6MTczMzA0Mjk1MH0.xw6FACf_lYmtjjvATEbCevOt-V8Cq-afuEJbT4OO34I")

import tradingeconomics as te
te.login('3e69f3322ec04da:riqejdprpxb0e5c')
calendar = te.getCalendarData(output_type='df')
set(calendar['Country'])