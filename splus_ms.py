#test new
import time, glob, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.table import Table
from numpy import random
from gatspy import datasets, periodic
from astropy.time import Time
from astropy.io import ascii
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from scipy.stats import kde
from pylab import *
from numpy import random
from astropy.table import Table
from numpy import random
from gatspy import datasets, periodic
from astropy.time import Time
from astropy.io import ascii
import matplotlib.patches as mpatches
from matplotlib.ticker import PercentFormatter
from scipy.stats import kde
from pylab import *
from numpy import random
from astropy.timeseries import LombScargle
from scipy.signal import savgol_filter
from astroML.time_series import search_frequencies, lomb_scargle, MultiTermFit

def lcplot_SPLUS_MS(tu,mag,emg,magc,per,deth,filtersu,filtu,path,ids,keyc):
	# definitions for the axes
	left, width = 0.09, 0.55; bottom, height = 0.10, 0.35
	bottom_h = left_h = left+width+0.02
	sz_box0 = [left,0.62, 0.89,0.32]
	sz_box1 = [left, bottom, width, height]
	sz_box2 = [left_h, bottom, 0.32, height]
	ac = sns.color_palette("tab20")
	fig = plt.figure(figsize=(8,5))
	plt.rcParams.update({'font.size': 12})
	# PLOT 01
	apt = np.max(mag)-np.min(mag)
	aptc = np.max(magc)-np.min(magc)
	box0 = plt.axes(sz_box0)
	box0.set(xlabel='Time( d )', ylabel='Magnitude',ylim=[np.max(mag)+0.1*apt,np.min(mag)-0.1*apt])
	box1 = plt.axes(sz_box1)
	box1.set(xlabel='Time( min )', ylabel='Norm. Magnitude', title='ID: '+ids,ylim=[np.max(magc)+0.2*aptc,np.min(magc)-0.2*aptc])
	box2 = plt.axes(sz_box2)
	box2.yaxis.set_major_formatter(plt.NullFormatter())
	box2.set(xlabel='Phase',  title='P: '+str(np.around(per[0]*60*24,2))+' min',ylim=[np.max(magc)+0.2*aptc,np.min(magc)-0.2*aptc])
	box2.yaxis.set_major_formatter(plt.NullFormatter())
	ph = (tu +  deth) /per % 1
	xx = np.concatenate([ph,1+ph]); yy = np.concatenate([magc,magc])
	ey = np.concatenate([emg,emg]); ff = np.concatenate([filtersu,filtersu])
	#print(tu)#print(mag)
	print(len(xx),len(yy),len(ey),len(ff),len(filtersu))
	for i in range(0,len(filtu)):
		box0.errorbar(tu[np.where(filtersu == filtu[i])], mag[np.where(filtersu== filtu[i])], yerr=emg[np.where(filtersu== filtu[i])], fmt='o', color=ac[i],ecolor='grey', elinewidth=1, capsize=4)
		box1.errorbar(tu[np.where(filtersu == filtu[i])]*24*60, magc[np.where(filtersu== filtu[i])], yerr=emg[np.where(filtersu== filtu[i])], fmt='o', color=ac[i],ecolor='black', elinewidth=1, capsize=4)
		box2.errorbar(xx[np.where(ff == filtu[i])], yy[np.where(ff== filtu[i])], yerr= ey[np.where(ff== filtu[i])], fmt='o', color='black',ecolor='black', elinewidth=1, capsize=4)
		box0.text(np.min(tu)+((np.max(tu)-np.min(tu))/len(filtu))*i,np.min(mag)-0.15*apt,filtu[i],color=ac[i])
	mtf.fit((tu +  deth), magc, emg)
	phase_fit, y_fit, phased_t = mtf.predict(1000, return_phased_times=True)
	box2.plot(phase_fit, y_fit, 'b', markeredgecolor='b', lw=4, fillstyle='top', linestyle='solid',color='lightblue')
	box2.plot(phase_fit+1, y_fit, 'b', markeredgecolor='b', lw=4, fillstyle='top', linestyle='solid',color='lightblue')
	fig.savefig(path+'/'+ids+keyc+'.png')
	return apt



def lcplot_SPLUS_TOO(tu,mag,emg,magc,per,deth,filtersu,filtu,path,ids):
	ph = (tu + deth) /per % 1
	xx = np.concatenate([ph,1+ph]); yy = np.concatenate([mag,mag])
	eyy = np.concatenate([emg,emg])
	mtf = MultiTermFit(2 * np.pi/per[0], 6)
	mtf.fit(tu+deth, mag, emg)
	phase_fit, y_fit, phased_t = mtf.predict(1000, return_phased_times=True)
	amp = np.max(y_fit) - np.min(y_fit)
	# definitions for the axes
	left, width = 0.09, 0.55; bottom, height = 0.15, 0.75
	bottom_h = left_h = left+width+0.02
	sz_box1 = [left, bottom, width, height]
	sz_box2 = [left_h, bottom, 0.32, height]
	# CHANGING FIG SIZE
	fig = plt.figure(figsize=(8,3))
	plt.rcParams.update({'font.size': 12})
	apt = np.max(mag)-np.min(mag)
	box1 = plt.axes(sz_box1)
	box1.set(xlabel='Time( min )', ylabel='Norm. Flux', title='ID: '+ids)
	box2 = plt.axes(sz_box2)
	box2.yaxis.set_major_formatter(plt.NullFormatter())
	box2.set(xlabel='Phase',  title='P: '+str(np.around(per[0]*60*24,2))+' min')
	box2.yaxis.set_major_formatter(plt.NullFormatter())
	box1.errorbar(tu, mag, yerr= emg, fmt='o', color='black',ecolor='black', elinewidth=1, capsize=4)
	box2.errorbar(xx, yy, yerr= eyy, fmt='o', color='black',ecolor='black', elinewidth=1, capsize=4)
	box2.plot(phase_fit, y_fit, 'b', markeredgecolor='b', lw=4, fillstyle='top', linestyle='solid',color='lightblue')
	box2.plot(phase_fit+1, y_fit, 'b', markeredgecolor='b', lw=4, fillstyle='top', linestyle='solid',color='lightblue')
	fig.savefig(path+'/'+ids+'_FollowUP_'+tile+'.png')



def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#============================================================
#GENERAL DEFINITIONS
#============================================================
font = {'family' : 'normal','weight' : 'bold','size'   : 24}
plt.style.use('seaborn-white')
path_data         = '/home/carlos/Documents/work_files/asplusV02/MS_SPLUS_data/'
path_simulations  = '/home/carlos/Documents/work_files/asplusV02/simulation/'
path_results      = '/home/carlos/Documents/work_files/asplusV02/CFL_PeriodAnalysis/'

#READIBG SIMULATION PARAMETERS
dts   = Table.read(path_simulations+'SimulationResultsSin_VANDERPLAS.dat', format='ascii')
dtn   = Table.read(path_simulations+'SimulationResultsNoise_UNIFORM_VANDERPLAS.dat', format='ascii')
dts   = dts[np.where(dts['Period']*24*60 < 40)]
dtn   = dtn[np.where(dtn['Period']*24*60 < 40)]
n0 = np.where(dts['SNR'] == 10)
n1 = np.where(dts['SNR'] == 5)
n2 = np.where(dts['SNR'] == 2)
li,lc = polyfit(dts['Period'][n2]*60*24, dts['Power01'][n2],1)

#============================================================
#ALL PROCESS - MAIN SURVEY - ANALYSIS AND FIGURES - MUTIPLE FOLDER
#============================================================
#HYDRA-0014  HYDRA-0119  SPLUS-n19s29  SPLUS-s37s34
hds = ['#SPLUS-ID','RA','Dec','Period(d)','Amplitude(mag)','Total_Time','Power','SNR1','SNR2','FlagVarType','NumDec','NumDecUsed']
ftu = ['J0395', 'J0410', 'J0430', 'J0515', 'J0861','gSDSS', 'iSDSS', 'rSDSS', 'zSDSS']
hdshift1 = ['J0378', 'J0395', 'J0410', 'J0430', 'J0515', 'J0660', 'J0861','gSDSS', 'iSDSS', 'rSDSS', 'uJAVA', 'zSDSS']
hdshift2 = np.array([220, 118, 59, 57, 61, 290, 80, 33, 46, 40, 227, 56])/(2.*24*60*60)

ff = glob.glob(path_data+'HYDRA*')
datatype = 'MS'
for itile in range(0,len(ff)):
	tile = os.path.basename(ff[itile])
	path_tile_results = path_results+tile+'_'+datatype+'_cflresults'
	os.mkdir(path_tile_results)
	#READING DATA FOR MAIN SURVEY
	dtms = Table.read(glob.glob(path_data+tile+'/*_out_mag.ascii')[0], format='ascii')
	dttt = Table.read(glob.glob(path_data+tile+'/*_fileList.ascii')[0], format='ascii')
	#SORTING HEADERs FOR TIME AND MAG FILES
	hdu = np.array(dtms.colnames); hdmag =[]
	for i in range(0,len(hdu)):
		x = hdu[i].split('_')
		for j in range(len(x),3):
			x.extend('-')
		hdmag.append(x)
	hdmag  = np.squeeze(hdmag)
	hdtime = (np.array(dttt['filter']))
	hduni  = np.unique(hdtime)
	#GETTING COMPARISSON LIGHT CURVE AND REMOVING BRIGHTER SOURCES
	flag = np.array(double(dtms[hdu[len(hdu)-1]]))
	for i in range(2,38):
		nn = np.where((double(dtms[hdmag[i][0]+'_'+hdmag[i][1]+'_'+hdmag[i][2]]) < 0) | (double(dtms['F_'+hdmag[i][1]+'_'+hdmag[i][2]]) != 0) | (flag < 12) | (flag > 20))
		if len(nn[0]) > 0:
			flag[nn] = -1
	dtmc = dtms[(flag > 0)]	
	#HEADER FOR MAGNITUDE VALUES
	hdall = np.concatenate([[hdmag[i][1]]*3 for i in range(2,38,3)])
	hdm = [hdmag[i][1] for i in range(2,38,3)]
	hda = [None]*(2*len(hdm))
	hda[::2] = hdm
	hda[1::2] = [hdm[i]+'_err' for i in range(0,len(hdm))]
	#READING TIME FILES
	rr = np.array([dttt['oa_date'],dttt['oa_time']]).T
	t = np.array([0.]*len(dttt)).T
	for k in range(0,len(dttt)):
		t[k] = Time(rr[k,0]+' '+rr[k,1]).mjd	
	t0 = t
	t = t-np.min(t)
	t = np.concatenate([t[i:i+3]+hdshift2[hdshift1.index(hdtime[i])] for i in range(0,len(hdtime),3)])
	for ik in range(0,5):		
		vstd = []
		print(tile,ik,len(dtmc))
		for iline in range(0,len(dtmc)):
			flx = np.array([-2.5*np.log10(dtmc[iline][hdu[k]]/np.mean(dtmc[hdu[k]])) for k in range(36*0+2,36*1+2)])
			vstd.append(np.std(flx))
		dtmc = dtmc[np.where(abs(vstd-np.mean(vstd)) < 2.5*np.std(vstd) )]
	vrr = []
	for iline in range(0,len(dtms)):
		rdc = np.array((np.array(dtms[iline][['ALPHA_J2000','DELTA_J2000']])).item())
		ids =  str('%.8f' % rdc[0])+ str('%.8f' % rdc[1])
		mg0 = np.array([-2.5*np.log10(dtms[iline][hdu[k]]) + 22.7 if dtms[iline][hdu[k]] > 0 else -999 for k in range(36*0+2,36*1+2)])
		mgm=np.concatenate([['%.4f' % np.mean(mg0[i:i+3]),'%.4f' % np.std(mg0[i:i+3])] if (mg0[i:i+3] > -8).all() else [-999]*2 for i in range(0,36,3)])
		mag = np.array([-2.5*np.log10(dtms[iline][hdu[k]]/np.mean(dtmc[hdu[k]])) if dtms[iline][hdu[k]] > 0 else -999 for k in range(36*0+2,36*1+2)])
		emg = np.array([ (1.5/log(10)) * ( dtms[iline][hdu[k+36]] / dtms[iline][hdu[k]] ) if dtms[iline][hdu[k]] > 0 else -999 for k in range(36*0+2,36*1+2)])
		fmg = np.array([dtms[iline][hdu[k]] for k in range(36*2+2,36*3+2)])		
		if len(np.where( (mag > 0) & (emg > 0) & (fmg == 0))[0]) > 10:
			keyh = [hdmag[i+2][1] if np.array([(mag[i:i+3] > -8).all(),(emg[i:i+3] > -8).all(),(fmg[i:i+3] == 0).all()]).all() else 0 for i in range(0,36,3)]
			tmps2 = [-999,-999,np.max(t),-999,-999,-999,-999]
			tmps2 = ['%.0f' % tmps2[i] if i != 2 else  '%.4f' % tmps2[i] for i in range(0,len(tmps2))]
			ninte = len(list(set(ftu).intersection(list(np.unique(keyh)))))
			if ninte > 6:
				#FINDING GOOD OBSERVATIONS - FLAG 0 AND VALID VALUES
				trr=np.concatenate([t[i:i+3] if np.array([(mag[i:i+3] > -8).all(),(emg[i:i+3] > 0).all(),(fmg[i:i+3] == 0).all()]).all() 
											else [-9]*3 for i in range(0,36,3)])
				mgr=np.concatenate([mag[i:i+3] if np.array([(mag[i:i+3] > -8).all(),(emg[i:i+3] > 0).all(),(fmg[i:i+3] == 0).all()]).all() 
											else [-9]*3 for i in range(0,36,3)])
				egr=np.concatenate([emg[i:i+3] if np.array([(mag[i:i+3] > -8).all(),(emg[i:i+3] > 0).all(),(fmg[i:i+3] == 0).all()]).all() 
											else [-9]*3 for i in range(0,36,3)])
				rrf=np.concatenate([mgr[i:i+3] if len(np.where(np.array(ftu) == hdtime[i])[0]) > 0 else [-9]*3 for i in range(0,len(hdtime),3)])>=-8
				#COMPUTING PERIODS
				model = periodic.LombScargleMultiband(fit_period=True,Nterms_base=1, Nterms_band=0)
				model.optimizer.period_range = (0.0035,(np.max(trr[rrf])-np.min(trr[rrf]))/2)
				model.fit(trr[rrf], mgr[rrf], egr[rrf], hdtime[rrf])
				power = model.periodogram(model.best_period)
				#COMPUTING "CORRECTED" FLUX - STD AMPLITUDE SNR
				t_fit = np.linspace(0, model.best_period, 1000)
				y_fit = model.predict(t_fit,hdtime[rrf][0])
				ymod  = model.predict(trr,hdtime[rrf][0])
				ymod = ymod - np.mean(y_fit)
				mgc  = np.concatenate([mgr[i:i+3]-(np.mean(mgr[i:i+3])-np.mean(ymod[i:i+3])) if mgr[i] > -8 else [-9]*3 for i in range(0,len(hdtime),3)])
				#IMPROVING PERIODS AND COMPUTE SOME CONSTRAINTS
				fls, pls = LombScargle(trr[rrf], mgc[rrf], egr[rrf]).autopower(minimum_frequency= 1/(model.best_period*1.1),
					maximum_frequency=1/(model.best_period*0.9),samples_per_peak=1000)
				per = 1./fls[np.where(pls == np.max(pls))]
				ph = trr[rrf]/per % 1
				ph = ph[np.argsort(trr[rrf]/per % 1)]
				ah = mgc[rrf]
				ah = smooth(ah[np.argsort(trr[rrf]/per % 1)],3)
				nh = np.where(ah == np.min(ah))
				mtf = MultiTermFit(2 * np.pi/per, 4)
				mtf.fit(trr[rrf]+per*(1-ph[nh[0]]), mgc[rrf], emg[rrf])
				phase_fit, y_fit, phased_t = mtf.predict(1000, return_phased_times=True)
				#SELECTING CANDIDATES
				limp = power - (li*model.best_period*60*24+lc)
				vart = 'Star' if limp <= 0 else 'WD/DS*'
				if limp >= 0:
					lcplot_SPLUS_MS(trr[rrf], mgr[rrf], egr[rrf],mgc[rrf],per,per[0]*(1-ph[nh[0]]),hdtime[rrf],np.unique(hdtime[rrf]),path_tile_results,ids,'_sel')
					lcplot_SPLUS_MS(trr[mgr>-5],mgr[mgr>-5],egr[mgr>-5],mgc[mgr>-5],per,per[0]*(1-ph[nh[0]]),hdtime[mgr>-5],np.unique(hdtime[mgr>-5]),path_tile_results,ids,'_all')
					vs = np.squeeze([t0[mgr>-5].T,mgr[mgr>-5].T,egr[mgr>-5].T,mgc[mgr>-5].T,hdtime[mgr>-5].T])
					vs = Table(vs.T,names=(['#Time(MJD)','Mag','err_Mag','Norm_Mag','Filters']),dtype=(['f','f','f','f','str']))
					ascii.write(vs,path_tile_results+'/'+ids+'_lc.dat', fill_values=[(ascii.masked, 'N/A')],overwrite=True)			
				amp = np.max(y_fit)-np.min(y_fit)
				tmps2 = [per[0], amp, np.max(t[rrf]), power, amp/np.std(emg[rrf]),  amp/np.std(mgc[rrf]-ymod[rrf])]
				tmps2 = ['%.8f' % tmps2[i] if i < 2 else  '%.4f' % tmps2[i] for i in range(0,len(tmps2))]
				tmps2.append(vart)
			tmps2 = np.concatenate([[ids],['%.8f'%rdc[0]],['%.8f'%rdc[1]],tmps2,[ninte*3],[len(np.where(mag>-8)[0])],mgm])
			vrr.append(tmps2)
			print(iline,len(tmps2),len(dtms))
	#SAVING TABLE
	vrr = Table(np.squeeze(vrr),names=np.concatenate([hds,hda]))
	ascii.write(vrr,path_tile_results+'/'+tile+'_CFL_PeriodAnalysis_'+datatype+'.dat', fill_values=[(ascii.masked, 'N/A')],overwrite=True)
	#MAKING FIGURES
	#***********************************
	#FIGURE POWER VERSUS PERIOD SELECT DATA
	#***********************************
	left, width = 0.075, 0.70
	bottom, height = 0.075, 0.675
	spacing = 0.005
	rect_scatter = [left, bottom, width, height]
	rect_histx = [left, bottom + height + spacing, width, 0.2]
	rect_histy = [left + width + spacing, bottom, 0.2, height]	
	xlimf = np.linspace(5,40,100)
	ylimf = li*xlimf+lc	
	# start with a square Figure
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_axes(rect_scatter,xlim=[4,38],ylim=[0., 1.05])
	ax_histx = fig.add_axes(rect_histx, sharex=ax)
	ax_histy = fig.add_axes(rect_histy, sharey=ax)
	ax.set(ylabel='Power',  xlabel='Period(min)')
	
	xy3, = ax.plot(dtn['Period']*60*24, dtn['Power'], 's', marker='.', color='grey', label='Noise', zorder=0)
	xy0, = ax.plot(dts['Period'][n0]*60*24, dts['Power01'][n0],linestyle='--', marker='o',color='gold' ,label='SNR = 10')
	ax.fill_between(dts['Period'][n0]*60*24, dts['Power01'][n0] -  dts['STD_Power01'][n0],dts['Power01'][n0] +  dts['STD_Power01'][n0], alpha=0.2,color='gold' )
	xy1, = ax.plot(dts['Period'][n1]*60*24, dts['Power01'][n1],linestyle='--', marker='o',color='red' ,label='SNR = 5')
	ax.fill_between(dts['Period'][n1]*60*24, dts['Power01'][n1] -  dts['STD_Power01'][n1],dts['Power01'][n1] +  dts['STD_Power01'][n1], alpha=0.2,color='red' )
	xy2, = ax.plot(dts['Period'][n2]*60*24, dts['Power01'][n2],linestyle='--', marker='o',color='green',label='SNR = 2')
	ax.fill_between(dts['Period'][n2]*60*24, dts['Power01'][n2] -  dts['STD_Power01'][n2],dts['Power01'][n2] +  dts['STD_Power01'][n2], alpha=0.2,color='green' )
	
	nsel = np.where( (double(vrr['Power'])-(li*double(vrr['Period(d)'])*60*24+lc)) > 0)
	nwdd = np.where(vrr['FlagVarType'] == 'WD/DS*')
	ax_histx.legend(handles=[xy0,xy1,xy2,xy3],loc=8, ncol=4,bbox_to_anchor=(0.45, 0.92),fontsize='small')
	xy5, = ax.plot(double(vrr['Period(d)'])*60*24, double(vrr['Power']),'o',color='black',label='MS-Splus', markersize=3)
	xy8, = ax.plot(double(vrr['Period(d)'][nwdd])*60*24, double(vrr['Power'][nwdd]),'o',color='green',label='MS-Splus', markersize=10)
	xy9, = ax.plot(double(vrr['Period(d)'][nsel])*60*24, double(vrr['Power'][nsel]),'o',color='black',label='MS-Splus', markersize=6)
	xy4, = ax.plot(double(vrr['Period(d)'][nsel])*60*24, double(vrr['Power'][nsel]),'o',color='cyan',label='MS-Splus', markersize=4)
	xy6, = ax.plot(xlimf,ylimf,linestyle='--',color='black',label='Model', markersize=4)
	# sorting parameters to x and y histograms
	ax_histx.tick_params(axis="x", labelbottom=False)
	ax_histy.tick_params(axis="y", labelleft=False)	
	x = dtn['Period']*60*24
	y = dtn['Power']
	# plot x y histogram
	for i in range(0,2):
		if i == 0:
			z = x
		else:
			z = y
		binr = (np.max(z)-np.min(z))/100
		lim = (int( (np.max(z)-np.min(z))/binr) + 1) * binr
		bins = np.arange(np.min(z)-binr, np.max(z) + binr, binr)
		if i == 0:
			ax_histx.hist(z, bins=bins, density=True,color='gray')
		else:
			ax_histy.hist(z, bins=bins, density=True, orientation='horizontal',color='gray')
	fig.savefig(path_tile_results+'/'+tile+'_Select_diagram.png')
	#***********************************
	#COLOR-COLOR DIAGRAM
	#***********************************
	x = double(vrr['rSDSS'])-double(vrr['iSDSS'])
	y = double(vrr['gSDSS'])-double(vrr['rSDSS'])
	rr = (double(vrr['rSDSS']) > 0) & (double(vrr['iSDSS']) > 0) & (double(vrr['gSDSS']) > 0) & (double(vrr['rSDSS']) > 0)
	x = x[rr]
	y = y[rr]
	nsel = np.where( (double(vrr['Power'][rr])-(li*double(vrr['Period(d)'][rr])*60*24+lc)) > 0)
	nwdd = np.where(vrr['FlagVarType'][rr] == 'WD/DS*')
	fig = plt.figure(figsize=(8,5))
	plt.rcParams.update({'font.size': 12})
	plt.plot(x, y, 'o', color='black', markersize=3)
	plt.xlabel('r - i')
	plt.ylabel('g - r')
	plt.title(tile)
	plt.plot(x[nwdd], y[nwdd], 'o', color='green', markersize=10)
	plt.plot(x[nsel], y[nsel], 'o', color='black', markersize=6)
	plt.plot(x[nsel], y[nsel], 'o', color='cyan', markersize=4)
	fig.savefig(path_tile_results+'/'+tile+'_Color_ColorDiagram_'+tile+'.png')	
	#***********************************
	#Period Amplitude cor
	#***********************************
	left, width = 0.09, 0.55; bottom, height = 0.10, 0.35
	bottom_h = left_h = left+width+0.02
	sz_box0 = [left,0.53, 0.89,0.42]
	sz_box1 = [left,0.10, 0.89,0.42]
	ac = sns.color_palette("tab20")
	nsel = np.where( ((double(vrr['Power'])-(li*double(vrr['Period(d)'])*60*24+lc)) > 0) & (double(vrr['Period(d)']) > 0)  & (double(vrr['gSDSS']) > 0))
	nwdd = np.where((vrr['FlagVarType'] == 'WD/DS*') & (double(vrr['Period(d)']) > 0))	
	fig = plt.figure(figsize=(8,5))
	plt.rcParams.update({'font.size': 12})
	box0 = plt.axes(sz_box0)
	box0.set(ylabel='g', title='Field: '+tile)
	#box1 = plt.axes(sz_box1,ylim=[0., 0.02])
	box1 = plt.axes(sz_box1)
	box1.set(xlabel='Period( min )', ylabel='Amplitude')
	box0.xaxis.set_major_formatter(plt.NullFormatter())
	xy0, = box0.plot(double(vrr['Period(d)'][nsel])*60*24, double(vrr['gSDSS'][nsel]),'s', marker='o',color='black')
	xy1, = box1.plot(double(vrr['Period(d)'][nsel])*60*24, double(vrr['Amplitude(mag)'][nsel]),'s', marker='o',color='black')
	fig.savefig(path_tile_results+'/'+tile+'PeriodAplitude'+tile+'.png')

