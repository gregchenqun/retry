##一般意义上的策略回测程序，包含一些策略评价参数的计算，收益率，连胜次数，
##夏普比例等，待做成一个打包的函数进行运用。
import numpy as np
import pandas as pd
import pylab as pl
import  matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\\windows\\fonts\\simsun.ttc", size=14) #画图中文字体设置


#读取excel数据，并原始数据初始化
data =pd.read_excel('D:\\index-d.xlsx')#读取excel，得到Dataframe
#data=data.reindex(range(data['开盘'].count()),method='ffill')#填充NAN数据
#data=data.sort_values(by=['时间'],ascending=False,inplace=True) #（可选）将日期由远及近排列
Date=data['时间']
Open=data['开盘']
High=data['最高']
Low=data['最低']
Close=data['收盘']

#资金面参数初始化
totalday=Open.size #总天数
length=Open.size  #总记录个数
total=np.zeros(length)    #总的资金额 ,49位数组
start_cash=500000      #初始资金
total[0:50]=start_cash #总资金额初始化，下标0-48 ，49个
position=0            #头寸仓位指示，1表示多头，-1表示空头
action_times=0        #操作次数
action_fail_times=0   #操作失败次数
charge=0              #交易费用
fee_rate=0.3/10000     #手续费
gain_total=0          #盈利累计
lost_total=0		#损失累计

#存储交易手数
openlong=0
openlongi=0
openshort=0
openshorti=0 
longpossess=0  #多头仓位
shortpossess=0 #空头仓位
position_change=0
#计算最大连亏和最大连赢
op=-1
operation=np.zeros(length)

#交易记录整理
recc=-1
trade_rec=pd.DataFrame(index=range(1000),columns=[u'日期',u'第几根K线',u'买卖价格',u'操作',u'资金',u'手续费'])#交易记录表
L_C=0 #多头的买入价格
S_C=0 #空头的买入价格

#开平仓标记
OPENLONG=0     #开多
OPENSHORT=0    #开空
CLEARLONG=0    #平多
CLEARSHORT=0   #平空

#策略参数预处理

TR=np.zeros(length)
HD=np.zeros(length)
LD=np.zeros(length)
DMP=np.zeros(length)
DMM=np.zeros(length)
PDI=np.zeros(length)
MDI=np.zeros(length)
MOV=np.zeros(length)

#波幅计算
for i in range(1,length):        #下标从1到（length-1）
   x =np.array([High[i]-Low[i],abs(High[i]-Close[i-1]),abs(Low[i]-Close[i-1])])
   TR[i]=max(x)

ATR=pd.rolling_mean(TR,14)
#rand=np.random.random(size=1)
for i in range(50,length):
#####产生交易指令信号############################################################

	if position==0:
		
		#rand=np.random.random(size=1)
		if Close[i-1]>max(Close[(i-40):i-1]): #价格大于前二十个的最大值，开多头
		#if rand>0.5:
			OPENLONG=1
		
		#if rand < 0.5: # 
		if Close[i-1]<min(Low[(i-20):i-1]):   #价格小于前十个的最小值，开空头
			OPENSHORT=1
	
	if position==1:  #多头平仓

		if L_C-Close[i-1]>3*ATR[i-1] or Close[i-1]<min(Low[(i-20):(i-1)]): #or (L_C-Open[i])/L_C>0.1 or (Open[i]-L_C)/L_C>0.2:# 单单只有止损
			CLEARLONG=1

	if position==-1: #空头平仓
		if Close[i-1]-S_C>3*ATR[i-1] or Close[i-1]>max(High[(i-40):(i-1)]):# or (Open[i]-S_C)/S_C>0.1 or (S_C-Open[i])/S_C>0.2:#同上
			CLEARSHORT=1
#####记录交易记录###################################################################
	if position==0:
		total[i]=total[i-1]
		if OPENLONG:  #空仓开多且仅在空仓时候开多
			position_change=1
			L_C=Open[i]   #多头交易价格
			op=op+1
			operation[op]=i#存贮所在K线的位置
			openlong=1#手数记录
			openlongi=i #位置记录
			longpossess=longpossess+1 #仓位变化
			total[i]=total[i-1]+openlong*(Close[i]-Open[i])*300-openlong*L_C*300*fee_rate#资金变化
			charge=charge+Open[i]*300*fee_rate #费用计算
			action_times=action_times+1 #开仓次数
			recc=recc+1 #交易记录表格的行数
			trade_rec.loc[recc,'日期']=Date[i]
			trade_rec.loc[recc,'第几根K线']=i#记录第几根K线
			trade_rec.loc[recc,'买卖价格']=Open[i]
			discription="开多"+str(openlong)+"手"
			trade_rec.loc[recc,'操作']=discription
			trade_rec.loc[recc,'资金']=total[i]
			trade_rec.loc[recc,'手续费']=Open[i]*300*fee_rate
			OPENLONG=0             #Information Received
			

		if OPENSHORT: #空仓开空
			position_change=-1
			S_C=Open[i]
			op=op+1
			operation[op]=i
			openshort=1
			openshorti=i
			shortpossess=shortpossess+1
			total[i]=total[i-1]-S_C*300*fee_rate
			total[i]=total[i]+openshort*(Open[i]-Close[i])*300
			charge=charge+Open[i]*300*fee_rate
			action_times=action_times+1
			recc=recc+1
			trade_rec.loc[recc,'日期']=Date[i]
			trade_rec.loc[recc,'第几根K线']=i#记录第几根K线
			trade_rec.loc[recc,'买卖价格']=S_C
			discription="开空"+str(openshort)+"手"
			trade_rec.loc[recc,'操作']=discription
			trade_rec.loc[recc,'资金']=total[i]
			trade_rec.loc[recc,'手续费']=Open[i]*300*fee_rate		
			OPENSHORT=0   #information received
			  #浮盈
				
	
	if position==1: ##多头平仓
		if CLEARLONG:
			position_change=-1
			op=op+1
			operation[op]=i
			total[i]=total[i-1]+longpossess*(Open[i]-Close[i-1])*300-longpossess*Open[i]*300*fee_rate
			recc=recc+1
			trade_rec.loc[recc,'日期']=Date[i]
			trade_rec.loc[recc,'第几根K线']=i#记录第几根K线
			trade_rec.loc[recc,'买卖价格']=Open[i]
			discription="平多"+str(longpossess)+"手"
			trade_rec.loc[recc,'操作']=discription
			trade_rec.loc[recc,'资金']=total[i]
			trade_rec.loc[recc,'手续费']=longpossess*(Open[i])*300*fee_rate
			charge=charge+longpossess*(Open[i])*300*fee_rate #交易费用
			longpossess=0

			if total[i]<total[openlongi]:
				lost_total=lost_total+total[openlongi]-total[i]
				action_fail_times=action_fail_times+1
			
			CLEARLONG=0
		else:
			total[i]=total[i-1]+longpossess*(Close[i]-Close[i-1])*300  #浮盈
	

	elif position==-1:  #空头平仓
		if CLEARSHORT:
			position_change=1
			op=op+1
			operation[op]=i
			total[i]= total[i-1]+ shortpossess*(Close[i-1]-Open[i])*300- shortpossess*Open[i]*300*fee_rate
			recc=recc+1
			trade_rec.loc[recc,'日期']=Date[i]
			trade_rec.loc[recc,'第几根K线']=i#记录第几根K线
			trade_rec.loc[recc,'买卖价格']=(Open[i])
			discription="平空"+str(shortpossess)+"手"
			trade_rec.loc[recc,'操作']=discription
			trade_rec.loc[recc,'资金']=total[i]
			trade_rec.loc[recc,'手续费']=shortpossess*(Open[i])*300*fee_rate
			charge=charge+shortpossess*(Open[i])*300*fee_rate
			shortpossess=0

			if total[i]<total[openshorti]:
				lost_total=lost_total+total[openshorti]-total[i]
				action_fail_times=action_fail_times+1

			CLEARSHORT=0
		else:
			total[i]=total[i-1]+shortpossess*(Close[i-1]-Close[i])*300   #不平仓，继续计算浮盈

	position=position+position_change
	position_change=0
	##交易结果记录#
trade_rec.to_excel("D:\\output.xls")


#计算最大回撤#####################################
max_retri=0
max_retri_ratio=0
retri_high_x=0 #产生最大回撤的高点位置
retri_high_y=0
retri_low_x=0  ##陈胜最大回撤的低点位置
retri_low_y=0

for i in range(1,length):
	delta=total[0:i]-np.tile(total[i],i)   ##向量运算代替第二层循环
	md=max(delta)
	if md>max_retri:
		max_retri=md   #z最大回撤数值
		retri_low_x=i
		retri_low_y=total[i]
		index_retri=max(max(np.where(delta==max_retri)))  #返回最大处的下标
		retri_high_x=index_retri
		retri_high_y=total[index_retri]
		max_retri_ratio=max_retri/retri_high_y
'''	for j in range(0,i):
		if total[j]-total[i]>max_retri:
			max_retri=total[j]-total[i]
			max_retri_ratio=max_retri/total[j]# 计算最大回撤值
			retri_high_x=j
			retri_high_y=total[j]
			retri_low_x=i
			retri_low_y=total[i]  ###得到高点和地点的坐标'''



##计算最大连亏##################
#print(operation[0:op])
len_op=operation[0:op+1].size    ###左闭右开注意
#print(len_op)
gain=np.zeros(int((len_op)/2))
#print(gain.size)
###计算得出收益数组####
for i in range(1,len_op):
	if i%2==0:
		gain[int(i/2)-1]=total[int(operation[i-1])]-total[int(operation[i-2])] ###因最终记录的次序为：开 平 开 平 。。。。。

print("gain")
#gain=np.delete(gain,gain.size-1,axis=0)##axis表示按行，消除gain数组的最后一个数字0
print(gain)
#print(total[int(operation[0]):-1])
print(gain.size)


#######计算最大连胜和最大连败次数#####################################
max_succ_win=0
win=0
max_succ_lose=0
lose=0
lose_series=np.zeros(gain.size)
win_series=np.zeros(gain.size)
 
for i in range(0,gain.size):
 	if gain[i]<0:
 		lose=lose+1
 		if i<gain.size-1:
	 		for j in range(i+1,gain.size):
	 			if gain[j]<0:
	 				lose=lose+1
	 			else:
	 				break

 		lose_series[i]=lose
 		lose=0  ##归零，进入下一个gain对象的计数周期
 		
 		#if lose>max_succ_lose:
 		#	max_succ_lose=lose
 		
 		#lose=0   
 
 	if gain[i]>0:
 		win=win+1
 		if i< gain.size-1:  ###下标小于最后一个位置是，执行向后的验证
	 		for j in range(i+1,gain.size):
	 			if gain[j]>0:
	 				win=win+1
	 			else:
	 				break

 		win_series[i]=win
 		win=0
 		#if win>max_succ_win:
 		#	max_succ_win=win
 		#win=0

max_succ_win=max(win_series)  ## 最大连胜
max_succ_lose=max(lose_series)  ###最大连败
print(lose_series)
print(win_series)


#######输出显示结果###################
#收益率
len=total.size
total_return=total[len-1]/total[0]-1    ##策略最终受益率
index_return=Close[length-1]/Close[0]-1
years=totalday/365                      ##总共的年数
annual_return=(total_return+1)**(1/years)-1  ##策略年化收益
index_annual_return=(index_return+1)**(1/years)-1  ##指数年化收益
print("操作次数：%d"%action_times)
print("策略受益：%f"%total_return)
print("指数收益：%f"%index_return)
#print("策略年化收益：%f"%annual_return)
#print("指数年化收益：%f"%index_annual_return)
print("最大回撤：%a"%max_retri_ratio)
print("最大回撤处：%d"%retri_low_x)
risk_free=0.035/365
returns=total
sharpe_ratio=(total_return-risk_free)/((gain/start_cash)).std()
print("夏普比例：%f"%sharpe_ratio)

success_ratio=1-action_fail_times/action_times
print("胜率：%f"%success_ratio)

print("期望收益：%f"%np.mean(gain))
print("最大连胜：%d"%max_succ_win)
print("最大连败：%d"%max_succ_lose)

#result_dic=pd.DataFrame({"策略收益:":[total_return],"指数收益:":[index_return],"最大回撤:":[max_retri_ratio],"最大回撤处:":[retri_low_x]})

#print(result_dic)


###作图###3

h=plt.figure(1)
plt.hist(lose_series,label='lose keep')
plt.title(u'连亏次数分布图',fontproperties='SimHei')
plt.xlabel('连亏次数',fontproperties='SimHei')
plt.ylabel('频率',fontproperties='SimHei')
#plt.show()
plt.grid(True)
plt.savefig('D:\\最大连亏次数统计.png',dpi=200)


h=plt.figure(2)
plt.plot(total)
plt.ylabel("Return")
plt.xlabel("Index")
#plt.lengend('该策略的收益',fontproperties='SimHei')
plt.title('资金净值图和最大回撤',fontproperties='SimHei')
plt.grid(True)
##标注最大回撤位置
plt.annotate(r'最大回撤'+str(max_retri_ratio), xy=(retri_low_x,retri_low_y),
            xycoords='data', xytext=(-90,-50),
            textcoords='offset points', fontsize=10,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),fontproperties='SimHei')
plt.savefig("D:\\净值和最大回撤.png",dpi=200)

#########################

h=plt.figure(3)
plt.subplot(2,1,1)
index=(start_cash/Close[0])*Close
plt.plot(index,color='red',linestyle='-')
plt.plot(total,color='green',linestyle='-') #跟踪指数收益图
plt.title("策略和指数收益对比图",fontproperties="SimHei")
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(total,color='green',linestyle='-')##策略受益图
plt.grid(True)
plt.annotate(r'最大回撤'+str(max_retri_ratio), xy=(retri_low_x,retri_low_y),
            xycoords='data', xytext=(-90,-50),
            textcoords='offset points', fontsize=10,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),fontproperties='SimHei')
plt.savefig("D:\\指数和策略对比图.png",dpi=200)

plt.show()
