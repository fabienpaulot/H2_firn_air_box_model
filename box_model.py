import numpy as np
from   numpy import cos,sin,pi,deg2rad,shape,append,isnan,array,zeros,nan,ones,min,abs,max,mean
from   numpy.matlib import repmat
from   numpy.linalg import inv,solve
from   scipy.linalg import block_diag
import glob as glob
import pandas as pd
import sys, getopt, os
import matplotlib.pyplot as plt
import copy
import json, pickle
from   joblib import Parallel,delayed
from   os import environ

g_mol = {}
g_mol['H2']  = 2.016  #grams/mol

gcol_order = np.array(['lat_90_60','lat_60_30','lat_30_0','lat_0_-30','lat_-30_-60','lat_-60_-90','strat_90_0','strat_0_-90'])

special_regions = {"nh":np.array(['lat_90_60','lat_60_30','lat_30_0']),
                   "nh_mt":np.array(['lat_60_30','lat_30_0']),
                   "sh":np.array(['lat_0_-30','lat_-30_-60','lat_-60_-90']),
                   "sh_mid_high":np.array(['lat_-30_-60','lat_-60_-90']),
                   "nh_mid_high":np.array(['lat_60_30','lat_90_60']),
                   "tropics":np.array(['lat_0_-30','lat_30_0']),
                   "all":gcol_order}

special_columns = {}

SA = 4*pi*(6378100**2) #total surface area of earth 
a1 = 2*pi*(6378100**2)*(sin(deg2rad(90))-sin(deg2rad(60))) #surface area in latitude bands on a sphere
a2 = 2*pi*(6378100**2)*(sin(deg2rad(60))-sin(deg2rad(30)))
a3 = 2*pi*(6378100**2)*(sin(deg2rad(30))-sin(0))
atm_SA=array([a1,a2,a3,a3,a2,a1])
areas=atm_SA/SA#fractional,surface,areas
frac_land = array([0.519,0.501,.287,0.23,0.059,0.19])
area_land = frac_land*areas
frac_geo  = area_land/np.sum(area_land)

abs_change_tol = 1e-5
rel_change_tol = 1e-5

HLex_strat=.046/.5;
fac_strat =1.#0.026242832543238
scale_strat=1#22.51528

Fo_default = array([0.25, 0.25, 0.9,0.9,0.6,0.6,0.9,0.9,0.6,0.6,
                     HLex_strat*areas[0]/(areas[0]+areas[1]),
                     HLex_strat*areas[0]/(areas[0]+areas[1]),
                     HLex_strat*areas[1]/(areas[0]+areas[1]),
                     HLex_strat*areas[1]/(areas[0]+areas[1]),
                     .018/.5,
                     .018/.5,
                     .018/.5,
                     .018/.5,
                     HLex_strat*areas[1]/(areas[0]+areas[1]),
                     HLex_strat*areas[1]/(areas[0]+areas[1]),
                     HLex_strat*areas[0]/(areas[0]+areas[1]),
                     HLex_strat*areas[0]/(areas[0]+areas[1]),
                     .027/1.5,.027/1.5])

delta_geo_default  =    0
delta_bb_default   = -290
delta_ant_default  = -196
delta_nfix_default = -628
delta_chem_default =  130 #16#Price (Pieterse is 116)
alpha_dep_default  =  0.925 #Pieterse
alpha_chem_default =  0.542

def save_output(out_ppb,FORCING,save_dir=None,output_dir=None,col_order=None,**kwargs):
        
    if save_dir is None:
        save_dir = output_dir
        
    for v in out_ppb:
        if len(out_ppb[v][:,0])>len(FORCING['time']):
            h2_avg = (out_ppb[v][1:,:]+out_ppb[v][:-1,:])/2
        else:
            h2_avg = out_ppb[v]

        pd.DataFrame(data=h2_avg,columns=col_order,index=FORCING['time']).to_csv(os.path.join(save_dir,v+".csv"))
        pd.DataFrame(data=calc_annual(h2_avg,FORCING,v,**kwargs),columns=col_order,index=FORCING['time'].year[::12]).to_csv(os.path.join(save_dir,v+"_annual.csv"))

def calc_annual(c,FORCING,source='H2',annual_avg='calendar',**kwargs):

    month_weight = FORCING['month_weight']
    if 'sink_chem' in source or 'sink_dep' in source or 'source' in source:
        c_annual = np.sum(np.reshape(c,[int(len(c[:,0])/12),12,len(c[0,:])]),axis=1)
    else:
        #calculate sum with no weight
        c_annual = np.sum(np.reshape(c,[int(len(c[:,0])/12),12,len(c[0,:])])*month_weight[:,:,None],axis=1)

    if annual_avg=='firn_smooth':        
        c_out=calc_firn_smoothing(c_annual,FORCING,**kwargs)
    else:
        c_out=c_annual
        
    return(c_out)

def calc_firn_smoothing(c,FORCING,trop=False,**kwargs):

    time = pd.to_datetime(FORCING['time'].year[::12],format='%Y')

    iobs={}
    imodel={}
    region = ['greenland','antarctica']
    for r in region:
        _,iobs[r],imodel[r] = np.intersect1d(FORCING[r].index,time,return_indices=True)

    col = {}
    col['greenland'] = 0
    if trop:
        col['antarctica']=-1
    else:
        col['antarctica']=-3

    h2 = {}
    for r in region:
        h2[r]=c[:(imodel[r][-1]+1),col[r]][::-1]
        
    REV_S={}
    for r in region:
        REV_S[r]=FORCING[r].to_numpy()[:,0:len(h2[r])]

    smoothed={}
    for r in region:
        REV_S[r]=REV_S[r]/np.sum(REV_S[r],1)[:,np.newaxis]
        smoothed[r]=REV_S[r].dot(h2[r])            

    co = copy.deepcopy(c)
    for r in region:
        co[imodel[r],col[r]] = smoothed[r][iobs[r]]

    return(co)
        

def convert_scale(scale,time,col_order,reference=None):

    if isinstance(scale,str):
        #read file
        scale = pd.read_csv(scale,parse_dates=['time'])

    if isinstance(scale,pd.DataFrame):
        #year vs month
        cs      = scale.reset_index()
        year    = cs['time'].dt.year

        cs = cs.set_index('time')
        TT = cs

        if len(year)>1 and int(len(np.unique(year)))==12:
            TT = pd.concat([cs]*int(len(time)/12))
            TT.index = time
        elif len(year)==1:
            TT = pd.concat([cs]*int(len(time)))
            TT.index = time
        elif len(year)>1 and int(len(np.unique(year)))!=int(len(year)/12):
            print("annual")
            #the scaling is annual
            #cs = repmat(cs.values,[12,a,b])
            print(cs)
            cs_s = cs.iloc[[-1]].shift(365,freq='D')
            cs   = pd.concat([cs,cs_s],axis=0)
            C    = cs.resample('1M').mean().ffill().dropna().shift(-15,freq='D')
            T    = pd.DataFrame(index=time)
            TT   = pd.concat([C,T],axis=1).interpolate(method='nearest').loc[time].fillna(1.)

        if TT.columns[0] == "all":
            TA = TT['all']
            for c in col_order:
                TT[c] = TA
        else:
            for c in col_order:
                is_special = False
                for s in special_regions:
                    if (TT.columns[0] == s and c in special_regions[s]):
                        print(c+"<-"+s)
                        TT[c] = TT[s]
                        is_special = True
                if (is_special)==False:
                    if (c in TT)==False:
                        TT[c] = 1.
                    
    else:
        TT = pd.DataFrame(index=time)
        if hasattr(scale,"__len__"):
            count=0
            for c in col_order:
                TT[c] = scale[count]
                count+=1
        else:
            for c in col_order:
                TT[c] = scale
            
    return(TT[col_order])
    
            

def read_forcing(input_dir="input",budget_dir="",budget_file="budget_ptp_patterson.csv",scale_src=1.,scale_bb=1.,scale_ant=1.,scale_soil=1.,scale_ocn=1.,scale_chem=1.,scale_geo=1.,col_order=gcol_order,scale_tau_dep=1.,scale_tau_chem=1.,
                 Fo          = Fo_default,
                 A1          = array([0.356,0.4,0.177,0.4,0.144,0.5,0.75,0.5,0.5,0.38,0.38,0.5 ]),
                 A2          = array([0.404,0.6,0.25,0.4,0.311,0,0,0,0,0,0,0]),                
                 scale_Fo = 1.,scale_A1=1.,scale_A2=1.,
                 scale_Fo_ant = 1.,scale_Fo_bb=1.,scale_Fo_chem=1.,scale_Fo_ocn=1.,scale_Fo_soil=1.,scale_Fo_geo=1.,
                 scale_A1_ant = 1.,scale_A1_bb=1.,scale_A1_chem=1.,scale_A1_ocn=1.,scale_A1_soil=1.,scale_A1_geo=1.,
                 scale_A2_ant = 1.,scale_A2_bb=1.,scale_A2_chem=1.,scale_A2_ocn=1.,scale_A2_soil=1.,scale_A2_geo=1.,
                 scale_tau_dep_ant=1.,scale_tau_dep_bb=1.,scale_tau_dep_ocn=1.,scale_tau_dep_chem=1.,scale_tau_dep_soil=1.,
                 geo_source=0.,lifetime_option="default",days_in_month="calendar",annual_avg="calendar",correction=None,firn_smoothing_version='V2',compound="H2",forcing_file=None,**kwargs):

    if forcing_file:

        with open(forcing_file, 'rb') as f:
            FORCING = pickle.load(f)
        
        if 'correction' in FORCING:
            del(FORCING['correction'])

        GLOBAL_MODEL_INPUT = None
            
    else:

        #SOURCES
        GLOBAL_MODEL_INPUT = pd.read_csv(os.path.join(input_dir,budget_dir,budget_file), parse_dates=['time'])
        GLOBAL_MODEL_INPUT = GLOBAL_MODEL_INPUT.sort_values(by='time')

        time           = GLOBAL_MODEL_INPUT['time'].unique()
        time           = pd.to_datetime(time)

        if days_in_month=="constant":
            dm = 30*np.ones(len(time))
        else:
            dm =time.days_in_month

        FORCING = {}

        dm = dm[:,None]
        FORCING['dm'] = dm

        month_weight            = np.reshape(dm,[int(len(dm)/12),12])
        month_weight_sum        = month_weight.sum(axis=1)
        FORCING['month_weight'] = month_weight/month_weight_sum[:,None]
        FORCING['Mw']           = g_mol[compound]
        FORCING['compound']     = compound

        if isinstance(Fo,list):
            Fo = np.array(Fo)
        FORCING['Fo']                = Fo
        FORCING['A1']                = A1
        FORCING['A2']                = A2
        if len(col_order)==6:
            FORCING['Fo']             = FORCING['Fo'][0:5]
            FORCING['A1']             = FORCING['A1'][0:5]                
            FORCING['A2']             = FORCING['A2'][0:5]            

        if isinstance(scale_Fo,str):
            scale_Fo = np.squeeze(pd.read_csv(scale_Fo,parse_dates=['time']).set_index('time').to_numpy())

        if isinstance(scale_A1,str):
            scale_A1 = np.squeeze(pd.read_csv(scale_A1,parse_dates=['time']).set_index('time').to_numpy())
        if isinstance(scale_A2,str):
            scale_A2 = np.squeeze(pd.read_csv(scale_A2,parse_dates=['time']).set_index('time').to_numpy())

        FORCING['Fo']             = FORCING['Fo']*scale_Fo
        FORCING['A1']             = FORCING['A1']*scale_A1
        FORCING['A2']             = FORCING['A2']*scale_A2

        FORCING['AIRMASS']        = GLOBAL_MODEL_INPUT.pivot(values='airmass_dry',index='time',columns='box')[col_order].to_numpy()

        if compound=="H2":

            scale_src      = convert_scale(scale_src,time = time,col_order=col_order)
            scale_bb       = convert_scale(scale_bb,time = time,col_order=col_order)*scale_src
            scale_ant      = convert_scale(scale_ant,time = time,col_order=col_order)*scale_src
            scale_soil     = convert_scale(scale_soil,time = time,col_order=col_order)*scale_src
            scale_ocn      = convert_scale(scale_ocn,time = time,col_order=col_order)*scale_src
            scale_geo      = convert_scale(scale_geo,time = time,col_order=col_order)*scale_src
            scale_chem     = convert_scale(scale_chem,time = time,col_order=col_order)*scale_src
            scale_tau_dep  = convert_scale(scale_tau_dep,time = time,col_order=col_order)
            scale_tau_dep_ant   = convert_scale(scale_tau_dep_ant,time = time,col_order=col_order)
            scale_tau_dep_bb    = convert_scale(scale_tau_dep_bb,time = time,col_order=col_order)
            scale_tau_dep_chem  = convert_scale(scale_tau_dep_chem,time = time,col_order=col_order)
            scale_tau_dep_soil  = convert_scale(scale_tau_dep_soil,time = time,col_order=col_order)
            scale_tau_dep_ocn   = convert_scale(scale_tau_dep_ocn,time = time,col_order=col_order)
            scale_tau_chem = convert_scale(scale_tau_chem,time = time,col_order=col_order)

            if isinstance(scale_Fo_ant,str):
                scale_Fo_ant = np.squeeze(pd.read_csv(scale_Fo_ant,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_Fo_bb,str):
                scale_Fo_bb = np.squeeze(pd.read_csv(scale_Fo_bb,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_Fo_chem,str):
                scale_Fo_chem = np.squeeze(pd.read_csv(scale_Fo_chem,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_Fo_ocn,str):
                scale_Fo_ocn = np.squeeze(pd.read_csv(scale_Fo_ocn,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_Fo_soil,str):
                scale_Fo_soil = np.squeeze(pd.read_csv(scale_Fo_soil,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_Fo_geo,str):
                scale_Fo_geo = np.squeeze(pd.read_csv(scale_Fo_geo,parse_dates=['time']).set_index('time').to_numpy())

            if isinstance(scale_A1_ant,str):
                scale_A1_ant = np.squeeze(pd.read_csv(scale_A1_ant,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A1_bb,str):
                scale_A1_bb = np.squeeze(pd.read_csv(scale_A1_bb,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A1_chem,str):
                scale_A1_chem = np.squeeze(pd.read_csv(scale_A1_chem,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A1_ocn,str):
                scale_A1_ocn = np.squeeze(pd.read_csv(scale_A1_ocn,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A1_soil,str):
                scale_A1_soil = np.squeeze(pd.read_csv(scale_A1_soil,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A1_geo,str):
                scale_A1_geo = np.squeeze(pd.read_csv(scale_A1_geo,parse_dates=['time']).set_index('time').to_numpy())

            if isinstance(scale_A2_ant,str):
                scale_A2_ant = np.squeeze(pd.read_csv(scale_A2_ant,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A2_bb,str):
                scale_A2_bb = np.squeeze(pd.read_csv(scale_A2_bb,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A2_chem,str):
                scale_A2_chem = np.squeeze(pd.read_csv(scale_A2_chem,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A2_ocn,str):
                scale_A2_ocn = np.squeeze(pd.read_csv(scale_A2_ocn,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A2_soil,str):
                scale_A2_soil = np.squeeze(pd.read_csv(scale_A2_soil,parse_dates=['time']).set_index('time').to_numpy())
            if isinstance(scale_A2_geo,str):
                scale_A2_geo = np.squeeze(pd.read_csv(scale_A2_geo,parse_dates=['time']).set_index('time').to_numpy())

            FORCING['Fo_ant']         = FORCING['Fo']*scale_Fo_ant
            FORCING['Fo_bb']          = FORCING['Fo']*scale_Fo_bb
            FORCING['Fo_chem']        = FORCING['Fo']*scale_Fo_chem
            FORCING['Fo_soil']        = FORCING['Fo']*scale_Fo_soil
            FORCING['Fo_ocn']         = FORCING['Fo']*scale_Fo_ocn
            FORCING['Fo_geo']         = FORCING['Fo']*scale_Fo_geo

            FORCING['A1_ant']         = FORCING['A1']*scale_A1_ant
            FORCING['A1_bb']          = FORCING['A1']*scale_A1_bb
            FORCING['A1_chem']        = FORCING['A1']*scale_A1_chem
            FORCING['A1_soil']        = FORCING['A1']*scale_A1_soil
            FORCING['A1_ocn']         = FORCING['A1']*scale_A1_ocn
            FORCING['A1_geo']         = FORCING['A1']*scale_A1_geo

            FORCING['A2_ant']         = FORCING['A2']*scale_A2_ant
            FORCING['A2_bb']          = FORCING['A2']*scale_A2_bb
            FORCING['A2_chem']        = FORCING['A2']*scale_A2_chem
            FORCING['A2_soil']        = FORCING['A2']*scale_A2_soil
            FORCING['A2_ocn']         = FORCING['A2']*scale_A2_ocn
            FORCING['A2_geo']         = FORCING['A2']*scale_A2_geo

            FORCING['SOURCE_bb']      = GLOBAL_MODEL_INPUT.pivot(values='h2_bb_emis3d',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound] * scale_bb[col_order].to_numpy() #~TmolH2/month        

            FORCING['SOURCE_ant']     = GLOBAL_MODEL_INPUT.pivot(values='h2_ant_emis',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]  * scale_ant[col_order].to_numpy()
            FORCING['SOURCE_soil']    = GLOBAL_MODEL_INPUT.pivot(values='h2_soil_emis',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound] * scale_soil[col_order].to_numpy()
            FORCING['SOURCE_ocn']     = GLOBAL_MODEL_INPUT.pivot(values='h2_ocn_emis',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]  * scale_ocn[col_order].to_numpy()
            FORCING['chem_ch4']       = GLOBAL_MODEL_INPUT.pivot(values='h2_prod_ch4',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]  * scale_chem[col_order].to_numpy()
            FORCING['chem_nch4']      = GLOBAL_MODEL_INPUT.pivot(values='h2_prod_nch4',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound] * scale_chem[col_order].to_numpy()

            FORCING['SOURCE_chem']    = FORCING['chem_ch4']+FORCING['chem_nch4']

            FORCING['SOURCE_geo']     = FORCING['SOURCE_soil']*0.
            for c in range(6):
                FORCING['SOURCE_geo'][:,c] = frac_geo[c]*geo_source/365*FORCING['dm'][:,0] * 1/g_mol[compound]
                
            FORCING['SOURCE_geo'] = FORCING['SOURCE_geo'] * scale_geo[col_order].to_numpy()
            
            FORCING['SOIL_SINK']      = GLOBAL_MODEL_INPUT.pivot(values='H2_ddep',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]
            FORCING['CHEM_SINK']      = GLOBAL_MODEL_INPUT.pivot(values='h2_loss',index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]



            SUM_SOURCE = 0.
            for v in FORCING:
                if 'SOURCE' in v:
                    SUM_SOURCE  += FORCING[v]

            FORCING['SOURCE'] = SUM_SOURCE * scale_src[col_order].to_numpy()

            FORCING['SINK']           = -(FORCING['SOIL_SINK']+FORCING['CHEM_SINK'])

            BURDEN                    = GLOBAL_MODEL_INPUT.pivot(values='H2_burden',index='time',columns='box')[col_order]
            FORCING['time']           = BURDEN.index
            FORCING['BURDEN']         = BURDEN.to_numpy() * 1/g_mol[compound]

            FORCING['CONCENTRATION']  = FORCING['BURDEN']/FORCING['AIRMASS'] * 1e12 /(1e3/29) * 1e9    

            if lifetime_option=="steady-state":

                #THIS IS THE STEADY-STATE LIFETIME, WE NEED THE INSTANTANEOUS LIFETIME
                FORCING['LIFETIME_DEP']   = -BURDEN.to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='H2_ddep',index='time',columns='box').fillna(0.)[col_order].to_numpy() * scale_tau_dep[col_order].to_numpy()
                FORCING['LIFETIME_CHEM']  = -BURDEN.to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='h2_loss',index='time',columns='box').fillna(0.)[col_order].to_numpy() * scale_tau_chem[col_order].to_numpy()

            else:

                FORCING['LIFETIME_CHEM']  = GLOBAL_MODEL_INPUT.pivot(values='lifetime_H2_OH',index='time',columns='box').fillna(1e20)[col_order].to_numpy() * scale_tau_chem[col_order].to_numpy() /(86400 * dm)

                FORCING['LIFETIME_DEP']   = GLOBAL_MODEL_INPUT.pivot(values='lifetime_dep',index='time',columns='box').fillna(1e20)[col_order].to_numpy() * scale_tau_dep[col_order].to_numpy() /(86400 * dm)

                #note that under these scenarios there is no deposition in Southern most box this can be a problem for inversion when using relative errors.
                #use instantaneous lifetime
                FORCING['LIFETIME_DEP'][:,5]   = -BURDEN['lat_-60_-90'].to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='H2_ddep',index='time',columns='box').fillna(0.)['lat_-60_-90'].to_numpy() * scale_tau_dep['lat_-60_-90'].to_numpy()

                count = 0
                for c in col_order:
                    if 'strat' in c:
                        FORCING['LIFETIME_CHEM'][:,count] = -(BURDEN.to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='h2_loss',index='time',columns='box').fillna(0.)[col_order].to_numpy() * scale_tau_chem[col_order].to_numpy())[:,count]
                    count+=1


            FORCING['LIFETIME_DEP']   = np.minimum(abs(FORCING['LIFETIME_DEP']),1e20)
            FORCING['LIFETIME_CHEM']  = np.minimum(abs(FORCING['LIFETIME_CHEM']),1e20)           

            #Tagged tracers

            s_tau_dep = {'ocn':scale_tau_dep_ocn,'bb':scale_tau_dep_bb,'soil':scale_tau_dep_soil,'ant':scale_tau_dep_ant,'chem':scale_tau_dep_chem}

            for v in ['bb','chem','ant','soil','ocn']:
                BURDEN                       = GLOBAL_MODEL_INPUT.pivot(values='H2_%s_burden'%v,index='time',columns='box')[col_order]

                if lifetime_option=="steady-state":
                    FORCING['LIFETIME_DEP_'+v]   = -BURDEN.to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='H2_%s_ddep'%v,index='time',columns='box').fillna(0.)[col_order].to_numpy() * s_tau_dep[v][col_order].to_numpy() * scale_tau_dep[col_order].to_numpy()
                    FORCING['LIFETIME_DEP_'+v]  = np.minimum(abs(FORCING['LIFETIME_DEP_'+v]),1e20)    
                    FORCING['LIFETIME_CHEM_'+v]  = -BURDEN.to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='h2_%s_loss'%v,index='time',columns='box').fillna(0.)[col_order].to_numpy() * scale_tau_chem[col_order].to_numpy()
                else:
                    FORCING['LIFETIME_CHEM_'+v]     = GLOBAL_MODEL_INPUT.pivot(values='lifetime_H2_'+v+'_OH',index='time',columns='box').fillna(1e20)[col_order].to_numpy() * scale_tau_chem[col_order].to_numpy() /(86400 * dm)
                    if "h2_weighted_dep_v1" in lifetime_option:            
                        FORCING['LIFETIME_DEP_'+v]  = GLOBAL_MODEL_INPUT.pivot(values='lifetime_dep_H2_'+v,index='time',columns='box').fillna(1e20)[col_order].to_numpy() / (86400 * dm) * s_tau_dep[v][col_order].to_numpy() * scale_tau_dep[col_order].to_numpy()
                    elif "h2_weighted_dep_v2" in lifetime_option:            
                        FORCING['LIFETIME_DEP_'+v]  = GLOBAL_MODEL_INPUT.pivot(values='lifetime_dep2_H2_'+v,index='time',columns='box').fillna(1e20)[col_order].to_numpy() / (86400 * dm) * s_tau_dep[v][col_order].to_numpy() * scale_tau_dep[col_order].to_numpy()
                    else:
                        FORCING['LIFETIME_DEP_'+v]  = GLOBAL_MODEL_INPUT.pivot(values='lifetime_dep',index='time',columns='box').fillna(1e20)[col_order].to_numpy() * scale_tau_dep[col_order].to_numpy() /(86400 * dm) * s_tau_dep[v][col_order].to_numpy()

                    #note that under these scenarios there is no deposition in Southern most box this can be a problem for inversion when using relative errors.
                    FORCING['LIFETIME_DEP_'+v][:,5]   = -BURDEN['lat_-60_-90'].to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='H2_%s_ddep'%v,index='time',columns='box').fillna(0.)['lat_-60_-90'].to_numpy() * s_tau_dep[v]['lat_-60_-90'].to_numpy() * scale_tau_dep['lat_-60_-90'].to_numpy()

                    count = 0
                    for c in col_order:
                        if 'strat' in c:
                            FORCING['LIFETIME_CHEM_'+v][:,count] = -(BURDEN.to_numpy()/GLOBAL_MODEL_INPUT.pivot(values='h2_%s_loss'%v,index='time',columns='box').fillna(0.)[col_order].to_numpy() * scale_tau_chem[col_order].to_numpy())[:,count]
                        count+=1


                FORCING['LIFETIME_DEP_'+v]   = np.minimum(abs(FORCING['LIFETIME_DEP_'+v]),1e20)
                FORCING['LIFETIME_CHEM_'+v]  = np.minimum(abs(FORCING['LIFETIME_CHEM_'+v]),1e20)           

                FORCING['BURDEN_'+v]         = BURDEN.to_numpy() * 1/g_mol[compound]
                FORCING['CONCENTRATION_'+v]  = FORCING['BURDEN_'+v]/FORCING['AIRMASS'] * 1e12 /(1e3/29) * 1e9            

            
                FORCING['SOIL_SINK_%s'%v]      = GLOBAL_MODEL_INPUT.pivot(values='H2_%s_ddep'%v,index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]
                FORCING['CHEM_SINK_%s'%v]      = GLOBAL_MODEL_INPUT.pivot(values='h2_%s_loss'%v,index='time',columns='box').fillna(0.)[col_order].to_numpy() * 1/g_mol[compound]

                
            FORCING['LIFETIME_DEP_geo']   = FORCING['LIFETIME_DEP_soil'] 
            FORCING['LIFETIME_CHEM_geo']  = FORCING['LIFETIME_CHEM_soil'] 
            FORCING['CONCENTRATION_geo']  = FORCING['CONCENTRATION_soil'] * 0
            FORCING['BURDEN_geo']         = FORCING['BURDEN_soil'] * 0

            #NOTE THAT THIS IS NOT CORRECT BUT IT DOES NOT MATTER SINCE WE ONLY USE THIS FOR RATIO
            FORCING['SOIL_SINK_geo']      = FORCING['SOIL_SINK_soil']
            FORCING['CHEM_SINK_geo']      = FORCING['CHEM_SINK_soil']            



        #TO SPEED UP CALC FIRN SMOOTHING
        if annual_avg=="firn_smooth":
            FORCING['greenland']=pd.read_csv(os.path.join(input_dir,'Firn Smoothing/greenland_firn_smoothing_%s.csv'%firn_smoothing_version),parse_dates=['time'],index_col='time')
            FORCING['antarctica']=pd.read_csv(os.path.join(input_dir,'Firn Smoothing/antarctic_firn_smoothing_%s.csv'%firn_smoothing_version),parse_dates=['time'],index_col='time')

        #CORRECTION
        if (correction is None)==False:
            FORCING['correction'] = pd.read_csv(correction,parse_dates=['time']).set_index(['compounds','time'])
            
    return(FORCING,GLOBAL_MODEL_INPUT)

def run_model(FORCING,transport="monthly",M0=None,it_start=0,ntimes=None,col_order=None,trop=True,tag=False,steps_per_mnth=30,compound="H2",**kwargs):

    if tag:

        output = {}

        for vv in ['bb','chem','ant','soil','ocn','geo']:
            F = copy.deepcopy(FORCING)

            F['LIFETIME_DEP']  = F['LIFETIME_DEP_'+vv]
            F['LIFETIME_CHEM'] = F['LIFETIME_CHEM_'+vv]                        
            F['CONCENTRATION'] = F['CONCENTRATION_'+vv]
            F['SOURCE']        = F['SOURCE_'+vv]
            F['Fo']            = F['Fo_'+vv]
            F['A1']            = F['A1_'+vv]
            F['A2']            = F['A2_'+vv]

            if (M0 is None):
                #then M0 needs to be float
                M0_c = M0
            else:
                M0_c = M0[compound+'_half_'+vv]
                
            temp_output = run_model(F,transport=transport,
                              M0=M0_c,
                              it_start=it_start,ntimes=ntimes,
                                    col_order=col_order,trop=trop,tag=False,steps_per_mnth=steps_per_mnth,compound=compound,
                                    **kwargs)
            
            #THIS IS ONLY USED FOR OPTIMIZATION
            if ('correction' in FORCING):

                if ntimes is None:
                    ntimes  = len(FORCING['time'])-it_start

                C = FORCING['correction'].loc['H2_'+vv]
                count=0

                for cc in col_order:
                    if cc in C:
                        temp_output['H2'][:,count] = temp_output['H2'][:,count]*C[cc]
                        count+=1

            for vv2 in temp_output:
                if (vv2 in output)==False:
                    output[vv2] = 0.
                output[vv2]        = output[vv2]+temp_output[vv2]
#                if '_' in vv2:
#                    vvt = vv2.split('_')[0]+'_'+vv+'_'+'_'.join(vv2.split('_')[1:])
#               else:
#                    vvt = vv2+'_'+vv
                output[vv2+'_'+vv] = temp_output[vv2]

        return(output)

    else:
    
        #BOX DEFINITION
        dlat_box = 30 #define model boxes - width of each latitude box in model 
        lat_top  = np.arange(90,-90,-dlat_box)
        lat_bot  = np.arange(60,-120,-dlat_box)
        mol_atmo = 5.115443e+18/29e-3 #1.77*10**20 #moles of "air" in the atmosphere (air = 29g/mol)
        mol_trop = 0.8*mol_atmo #moles of "air" in the troposphere (up to 200 hPa, 80#)
        #mol_box = 0.8*mol_atmo*areas #moles air in boxes
        if trop:
            mol_box = mol_trop*areas #moles air in boxes
            mol_atmo = mol_trop
        else:
            mol_box = append(0.8*mol_atmo*areas,array([0.2*mol_atmo*0.5,0.2*mol_atmo*0.5])) #moles air in boxes

        #transport
        Fo          = FORCING['Fo']
        Fo_constant = repmat(Fo, 12,1)

        A1 = FORCING['A1']
        A2 = FORCING['A2']

        phi1        = array([0.068,0.451,0.848,0.451,0.75,\
                             0.5,0.5,0,0.5,0.25,0.25,0.5])
        phi2        = array([0.905,0.005,0.984,0.787,0.093,\
                             0.1,0,0,0.32,0.32,0.45,0.25])
        t           = np.arange(0.5,12.5)/12
        Ft          = zeros((len(Fo),len(t)))*nan
        
        indexer=[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11]

        if transport == 'monthly':
            for i in range(len(Fo)):
                Ft[i,:] = Fo[i]*(1 + A1[indexer[i]]*cos(2*pi*(t-phi1[indexer[i]])) + A2[indexer[i]]*cos(4*pi*(t-phi2[indexer[i]])))
        else:
            for i in range(len(Fo)):            
                Ft[i,:] = Fo_constant[i,:]

        #F0_1 is box1 (reversed from the EBAMM model where Box1 is the HSL)  # Fo = [0.25, 0.9, 0.6, 0.9, 1.2];
        gam = 177e18;# global air mass from Marik PhD
        k12 = (Ft[0,:]*mol_atmo)/mol_box[0] #scaling transport parameters based on box A volume --> box B volume (i.e. k12 = from box 1 to box 2)
        k21 = (Ft[1,:]*mol_atmo)/mol_box[1]
        k23 = (Ft[2,:]*mol_atmo)/mol_box[1]
        k32 = (Ft[3,:]*mol_atmo)/mol_box[2]
        k34 = (Ft[4,:]*mol_atmo)/mol_box[2]
        k43 = (Ft[5,:]*mol_atmo)/mol_box[3]
        k45 = (Ft[6,:]*mol_atmo)/mol_box[3]
        k54 = (Ft[7,:]*mol_atmo)/mol_box[4]
        k56 = (Ft[8,:]*mol_atmo)/mol_box[4]
        k65 = (Ft[9,:]*mol_atmo)/mol_box[5]

        if trop:
            ktr_6box    = array([k12, k21, k23, k32, k34, k43, k45, k54, k56, k65])/365; #transport is day^-1
            ktr_monthly = ktr_6box*365/12
            ktr_yearly  = ktr_monthly*12/1
        else:
            k17 = (Ft[10,:]*mol_atmo)/mol_box[0]  * fac_strat
            k71 = (Ft[11,:]*mol_atmo)/mol_box[6]  * fac_strat*scale_strat   
            k27 = (Ft[12,:]*mol_atmo)/mol_box[1]  * fac_strat
            k72 = (Ft[13,:]*mol_atmo)/mol_box[6]  * fac_strat*scale_strat    
            k37 = (Ft[14,:]*mol_atmo)/mol_box[2]  * fac_strat
            k73 = (Ft[15,:]*mol_atmo)/mol_box[6]  * fac_strat*scale_strat   

            k48 = (Ft[16,:]*mol_atmo)/mol_box[3]  * fac_strat
            k84 = (Ft[17,:]*mol_atmo)/mol_box[7]  * fac_strat*scale_strat   
            k58 = (Ft[18,:]*mol_atmo)/mol_box[4]  * fac_strat
            k85 = (Ft[19,:]*mol_atmo)/mol_box[7]  * fac_strat*scale_strat   
            k68 = (Ft[20,:]*mol_atmo)/mol_box[5] * fac_strat
            k86 = (Ft[21,:]*mol_atmo)/mol_box[7] * fac_strat*scale_strat

            k78 = (Ft[22,:]*mol_atmo)/mol_box[6]
            k87 = (Ft[23,:]*mol_atmo)/mol_box[7]    

            ktr_8box    = array([k12, k21, k23, k32, k34, k43, k45, k54, k56, k65,  #0-9
                                 k17, k71, k27, k72, k37, k73,                      #10-15
                                 k48, k84, k58, k85, k68, k86,                      #16-21
                                 k78, k87])/365; #transport is day^-1               #22-23

    #        ktr_8box[10:] = 0.
    #        ktr_8box = 0.8 * ktr_8box

            ktr_monthly = ktr_8box*365/12
            ktr_yearly  = ktr_monthly*12/1


        msteps         = len(FORCING['time'])
        years          = msteps/12
        M              = zeros([msteps+1,len(col_order)])

        k_dep = 1./FORCING['LIFETIME_DEP']  
        k_chem= 1./FORCING['LIFETIME_CHEM'] 

        if M0 is None:
            M[it_start,:]      = FORCING['CONCENTRATION'][it_start,:]
        else:
            M[it_start,:]      = M0

        M[it_start,:]         = M[it_start,:]*mol_box[None,:]/(1e9 * 1e12)
        
        if ntimes is None:
            ntimes  = len(FORCING['time'])-it_start

        #SOLVE

        sink_chem    = np.zeros([msteps,len(col_order)])
        sink_dep     = np.zeros([msteps,len(col_order)])

        sources_diag = np.zeros([msteps,len(col_order)])

        for jj in np.arange(it_start,it_start+ntimes):

            M_keep      = M[jj,:]
            source_mnth = FORCING['SOURCE'][jj,:]
            trn_k       = ktr_monthly[:,jj%12]


            if trop:
                A=array([[trn_k[0], -trn_k[1], 0, 0, 0, 0],
                         [-trn_k[0],  trn_k[1]+trn_k[2], -trn_k[3], 0, 0, 0],
                         [0, -trn_k[2], trn_k[3] + trn_k[4], -trn_k[5], 0, 0],
                         [0, 0, -trn_k[4],  trn_k[5] + trn_k[6], -trn_k[7], 0],
                        [0, 0, 0, -trn_k[6], trn_k[7] + trn_k[8], -trn_k[9]],
                         [0, 0, 0, 0, -trn_k[8],  trn_k[9]]])#transport matrix
            else:
                A=array([[trn_k[0]+trn_k[10], -trn_k[1], 0, 0, 0, 0, -trn_k[11], 0],
                         [-trn_k[0],  trn_k[1]+trn_k[2]+trn_k[12], -trn_k[3], 0, 0, 0, -trn_k[13] , 0],
                         [0, -trn_k[2], trn_k[3]+trn_k[4]+trn_k[14], -trn_k[5], 0, 0, -trn_k[15], 0],
                         [0, 0, -trn_k[4],  trn_k[5]+trn_k[6]+trn_k[16], -trn_k[7], 0, 0, -trn_k[17]],
                         [0, 0, 0, -trn_k[6], trn_k[7]+trn_k[8]+trn_k[18], -trn_k[9], 0 , -trn_k[19] ],
                         [0, 0, 0, 0, -trn_k[8],  trn_k[9]+trn_k[20], 0, -trn_k[21]],
                         [-trn_k[10], -trn_k[12], -trn_k[14], 0, 0, 0, trn_k[11]+trn_k[13]+trn_k[15]+trn_k[22], -trn_k[23]],
                         [0, 0, 0, -trn_k[16],-trn_k[18], -trn_k[20], -trn_k[22], trn_k[23]+trn_k[17]+trn_k[19]+trn_k[21]]])#transport matrix        



            for kk in np.arange(steps_per_mnth):
                sources            = source_mnth/steps_per_mnth
                sink               = M_keep*(k_chem[jj,:]+k_dep[jj,:])/steps_per_mnth
                sink_dep[jj,:]    -=M_keep*k_dep[jj,:]/steps_per_mnth
                sink_chem[jj,:]   -=M_keep*k_chem[jj,:]/steps_per_mnth
                sources_diag[jj,:]+=sources
                transport          = np.matmul(A/steps_per_mnth,M_keep)
                M_keep             = M_keep+sources-transport-sink                                

            M[jj+1,:]=M_keep


#        out_ppb=M[(it_start+1):(it_start+ntimes+1),:]/mol_box[None,:]*1e9 * 1e12
        out_ppb=M/mol_box[None,:]*1e9 * 1e12

        sink_dep     = sink_dep*FORCING['Mw']
        sink_chem    = sink_chem*FORCING['Mw']
        sources_diag = sources_diag*FORCING['Mw']
        burden_Tg = (M[1:,:]+M[:-1,:])/2*FORCING['Mw']

        conc = (out_ppb[1:,:]+out_ppb[:-1,:])/2 
        if (it_start>0):
            conc[0:it_start,:] = 0. #otherwise (it_start-1) is M0/2 for it_start>0

        return({compound:conc,compound+'_half':out_ppb,compound+'_burden':burden_Tg,compound+'_sink_dep':sink_dep,compound+'_sink_chem':sink_chem,compound+'_source':sources_diag})


def optimize_figure(model_ref,model_opt,observation,output_optimize,inversion_resolution="monthly",col_order=None,forcing=None,compound="H2",**kwargs):

    #for backward compatibility
    if type(observation) is dict:
        observation = np.array([observation])
    
    for io in range(len(observation)):
        if ('target' in observation[io])==False:
            target = compound
        else:
            target = observation[io]['target']
        
        #read observations
        obs = pd.read_csv(observation[io]['file'],parse_dates=[0]).set_index('time')
        obs = obs.dropna(how="all",axis=1)

        #get model
        h2_avg_ref = model_ref[target]
        h2_avg_opt = model_opt[target]

        H2_avg_ref = pd.DataFrame(data=h2_avg_ref,columns=col_order,index=forcing['time'])
        H2_avg_opt = pd.DataFrame(data=h2_avg_opt,columns=col_order,index=forcing['time'])    

        if len(obs.columns)==6:
            fig,ax=plt.subplots(3,2,sharex=True);ax=ax.flat
        elif len(obs.columns)==4:
            fig,ax=plt.subplots(2,2,sharex=True);ax=ax.flat            
        else:
            fig,ax=plt.subplots(len(obs.columns),1,sharex=True);ax=ax.flat
            
        count = 0
        for c in col_order:
            if c in obs:
                ax[count].plot(obs.index.to_pydatetime(),obs[c],linewidth=0.5,label='observation',color='black')#marker='o',markerfacecolor='None')
                ax[count].plot(H2_avg_ref.index,H2_avg_ref[c],linewidth=0.5,label='apriori')
                ax[count].plot(H2_avg_opt.index,H2_avg_opt[c],linewidth=0.5,label='optimized')
                ax[count].set_title(c)
                count+=1
        plt.legend()
        plt.savefig(output_optimize+"/%s_ts.pdf"%target)

        H2_avg_ref = pd.DataFrame(data=calc_annual(h2_avg_ref,forcing,target,**kwargs),columns=col_order,index=forcing['time'].year[::12])
        H2_avg_opt = pd.DataFrame(data=calc_annual(h2_avg_opt,forcing,target,**kwargs),columns=col_order,index=forcing['time'].year[::12])

        H2_avg_ref.index = pd.to_datetime(H2_avg_ref.index, format='%Y')
        H2_avg_opt.index = pd.to_datetime(H2_avg_opt.index, format='%Y')


        if len(obs.columns)==6:
            fig,ax=plt.subplots(3,2,sharex=True);ax=ax.flat
        elif len(obs.columns)==4:
            fig,ax=plt.subplots(2,2,sharex=True);ax=ax.flat            
        else:
            fig,ax=plt.subplots(len(obs.columns),1,sharex=True);ax=ax.flat
        
        count = 0
        if inversion_resolution=="monthly":
            obs['year'] = obs.index.year
            obs = obs.groupby('year').mean()
            obs.index = pd.to_datetime(obs.index,format="%Y")

        for c in col_order:
            if c in obs:
                ax[count].plot(obs.index.to_pydatetime(),obs[c],label='observation',color='black')#,marker='o',markerfacecolor='None')
                ax[count].plot(H2_avg_ref.index,H2_avg_ref[c],label='apriori')
                ax[count].plot(H2_avg_opt.index,H2_avg_opt[c],label='optimized')
                ax[count].set_title(c)            
                count+=1
        plt.legend()        
        plt.savefig(output_optimize+"/%s_ts_annual.pdf"%target)
        
def diagnostics_figure(GLOBAL_MODEL_INPUT,BOX_MODEL_OUTPUT,FORCING,col_order=gcol_order,input_dir="input",output_dir="output",
                       budget_dir="",compound="H2",**kwargs):

    tracer_suffix = ['','_bb','_chem','_ant','_soil','_ocn','_geo']

    #PRINT LIFETIME
    #use year 15-35
    for vt in tracer_suffix:

        try:            
            BURDEN     = pd.DataFrame(data=BOX_MODEL_OUTPUT['%s_burden'%compound+vt],columns=col_order,index=FORCING['time'])
            SINK_DEP   = pd.DataFrame(data=BOX_MODEL_OUTPUT['%s_sink_dep'%compound+vt],columns=col_order,index=FORCING['time'])                
            SINK_CHEM  = pd.DataFrame(data=BOX_MODEL_OUTPUT['%s_sink_chem'%compound+vt],columns=col_order,index=FORCING['time'])

            BURDEN        = BURDEN.loc[slice('1885-01-01','1900-12-31')]
            BURDEN['SUM'] = BURDEN.sum(axis=1)
            BURDEN['DAY_IN_MONTH'] = BURDEN.index.days_in_month
            SINK_DEP    = -SINK_DEP.loc[slice('1885-01-01','1900-12-31')].sum(axis=1)
            SINK_CHEM   = -SINK_CHEM.loc[slice('1885-01-01','1900-12-31')].sum(axis=1)

            BURDEN      = ((BURDEN['SUM']*BURDEN["DAY_IN_MONTH"]).groupby(BURDEN.index.year).sum()/BURDEN['DAY_IN_MONTH'].groupby(BURDEN.index.year).sum()).mean()
            SINK_DEP    = SINK_DEP.groupby(SINK_DEP.index.year).sum().mean()
            SINK_CHEM   = SINK_CHEM.groupby(SINK_CHEM.index.year).sum().mean()

            GBURDEN       = GLOBAL_MODEL_INPUT.pivot(values="H2%s_burden"%vt,index='time',columns='box')[col_order]
            GSINK_DEP     = -GLOBAL_MODEL_INPUT.pivot(values="H2%s_ddep"%vt,index='time',columns='box')[col_order]
            GSINK_CHEM    = -GLOBAL_MODEL_INPUT.pivot(values="h2%s_loss"%vt,index='time',columns='box')[col_order]

            GBURDEN       = GBURDEN.loc[slice('1885-01-01','1900-12-31')]
            GBURDEN['SUM'] = GBURDEN.sum(axis=1)
            GBURDEN['DAY_IN_MONTH'] = GBURDEN.index.days_in_month

            GBURDEN      = ((GBURDEN['SUM']*GBURDEN["DAY_IN_MONTH"]).groupby(GBURDEN.index.year).sum()/GBURDEN['DAY_IN_MONTH'].groupby(GBURDEN.index.year).sum()).mean()
            GSINK_DEP     = GSINK_DEP.loc[slice('1885-01-01','1900-12-31')].sum(axis=1)
            GSINK_CHEM    = GSINK_CHEM.loc[slice('1885-01-01','1900-12-31')].sum(axis=1)
            GSINK_DEP    = GSINK_DEP.groupby(GSINK_DEP.index.year).sum().mean()
            GSINK_CHEM   = GSINK_CHEM.groupby(GSINK_CHEM.index.year).sum().mean()

            print("")
            print(compound+vt)
            print("tau_dep=%3.2f tau_chem=%3.2f tau=%3.2f"%(BURDEN/SINK_DEP,BURDEN/SINK_CHEM,BURDEN/(SINK_DEP+SINK_CHEM)))
            print("gtau_dep=%3.2f gtau_chem=%3.2f gtau=%3.2f"%(GBURDEN/GSINK_DEP,GBURDEN/GSINK_CHEM,GBURDEN/(GSINK_DEP+GSINK_CHEM)))
            print("")
        except:
            pass

    for vt in tracer_suffix:

        v = compound+vt
        if v in BOX_MODEL_OUTPUT:

            h2_avg  = BOX_MODEL_OUTPUT[v]

            out_ppb = pd.DataFrame(data=h2_avg,columns=col_order,index=FORCING['time'])        
            fig,ax = plt.subplots(int(len(col_order)/2),2,figsize=(10,10),sharex=True,sharey=True);ax=ax.flat

            count  = 0

            if v+'_burden' in GLOBAL_MODEL_INPUT.columns:

                GMODEL_CONC = GLOBAL_MODEL_INPUT.pivot(values=v+'_burden',index='time',columns='box')[col_order]/GLOBAL_MODEL_INPUT.pivot(values='airmass_dry',index='time',columns='box')[col_order] * 1e12/FORCING['Mw']   * 1/(1e3/29) * 1e9
                try:
                    GMODEL_SCONC = GLOBAL_MODEL_INPUT.pivot(values=v+'_dvmr_sfc',index='time',columns='box')[col_order]*1e9
                except:
                    GMODEL_SCONC = {}

                ice_core_file = os.path.join(input_dir,budget_dir,'ice_core_%s_dvmr.csv'%v)

                if os.path.exists(ice_core_file):
                    I  = pd.read_csv(ice_core_file,parse_dates=[0]).sort_values(by='time').set_index('time')
                else:
                    I = {}

                for c in col_order:
                    out_ppb[c].rolling(12).mean().plot(ax=ax[count],label='box model')
                    GMODEL_CONC[c].rolling(12).mean().plot(ax=ax[count],label='global model')

#                    out_ppb[c].plot(ax=ax[count],label='box model',linewidth=0.5)
#                    GMODEL_CONC[c].plot(ax=ax[count],label='global model',linewidth=0.5)

                    if c in GMODEL_SCONC:
                        GMODEL_SCONC[c].rolling(12).mean().plot(ax=ax[count],label='global surface model',linestyle='--',linewidth=1)
                        if c == 'lat_90_60' and 'Summit' in I:
                            I['Summit'].rolling(12).mean().plot(ax=ax[count],label='Summit (global model)',linestyle='--',linewidth=1)
                        if c == 'lat_-60_-90' and 'South Pole' in I:
                            I['South Pole'].rolling(12).mean().plot(ax=ax[count],label='South Pole (global model)',linestyle='--',linewidth=1)
                    ax[count].legend()                    
                    ax[count].set_title(c)
                    count+=1

            plt.suptitle(v,fontsize=15)            
            plt.savefig(os.path.join(output_dir,v+"_ts.pdf"))

    plt.close('all')
    
    #sinks
    for vt in tracer_suffix:        

        v  = '%s_sink_dep'%compound+vt
        vg = compound+vt+'_ddep'

        count = 0        

        if v in BOX_MODEL_OUTPUT:

            if vg in GLOBAL_MODEL_INPUT.columns:

                fig,ax = plt.subplots(int(len(col_order)/2)+1,2,figsize=(10,10),sharex=True);ax=ax.flat                            
                P = GLOBAL_MODEL_INPUT.pivot(values=vg,index='time',columns='box')[col_order]

                out_box = pd.DataFrame(data=BOX_MODEL_OUTPUT[v],columns=col_order,index=FORCING['time'])                
                for c in col_order:

                    out_box[c].rolling(12).mean().plot(ax=ax[count],label='box model')
                    P[c].rolling(12).mean().plot(ax=ax[count],label='global model')

                    ax[count].legend()                    
                    ax[count].set_title(c)
                    count+=1

                out_box.sum(axis=1).rolling(12).mean().plot(ax=ax[count],label='box model')
                P.sum(axis=1).rolling(12).mean().plot(ax=ax[count],label='global model')
                    
                plt.suptitle(v,fontsize=15)            
                plt.savefig(os.path.join(output_dir,v+"_ts.pdf"))

                
        plt.close('all')
        
    for vt in tracer_suffix:        

        v  = '%s_sink_chem'%compound+vt
        vg = '%s'%compound.lower()+vt+'_loss'

        count = 0        

        if v in BOX_MODEL_OUTPUT:

            if vg in GLOBAL_MODEL_INPUT.columns:

                fig,ax = plt.subplots(int(len(col_order)/2),2,figsize=(10,10),sharex=True);ax=ax.flat                            
                P = GLOBAL_MODEL_INPUT.pivot(values=vg,index='time',columns='box')[col_order]

                out_box = pd.DataFrame(data=BOX_MODEL_OUTPUT[v],columns=col_order,index=FORCING['time'])                
                for c in col_order:

                    out_box[c].rolling(12).mean().plot(ax=ax[count],label='box model')
                    P[c].rolling(12).mean().plot(ax=ax[count],label='global model')

                    ax[count].legend()                    
                    ax[count].set_title(c)
                    count+=1

                plt.suptitle(v,fontsize=15)            
                plt.savefig(os.path.join(output_dir,v+"_ts.pdf"))

    plt.close('all')

    #source
    for s in ['SOURCE']:
        for vt in tracer_suffix:        

            v = '%s_source'%compound+vt
            vg = s+vt
            count = 0

            if v in BOX_MODEL_OUTPUT:

                if vg in FORCING:

                    fig,ax = plt.subplots(int(len(col_order)/2),2,figsize=(10,10),sharex=True);ax=ax.flat                            
                    out_global = pd.DataFrame(data=FORCING[vg],columns=col_order,index=FORCING['time'])*FORCING['Mw']
                    out_box    = pd.DataFrame(data=BOX_MODEL_OUTPUT[v],columns=col_order,index=FORCING['time'])                
                    for c in col_order:

                        out_box[c].rolling(12).mean().plot(ax=ax[count],label='box model')
                        out_global[c].rolling(12).mean().plot(ax=ax[count],label='global model')

                        out_box[c].plot(ax=ax[count],label='box model',linewidth=0.5)
                        out_global[c].plot(ax=ax[count],label='global model',linewidth=0.5)

                        ax[count].legend()                    
                        ax[count].set_title(c)
                        count+=1

                    plt.suptitle(v,fontsize=15)            
                    plt.savefig(os.path.join(output_dir,v+"_ts.pdf"))

    plt.close('all')
                    
def perturb_forcing(forcing,perturbation,scale):

    forcing_pert = copy.deepcopy(forcing)
        
    if 'tau_lifetime_dep_all' in perturbation:        
        for v in forcing:
            if 'LIFETIME_DEP' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)

    elif 'tau_lifetime_dep_ant' in perturbation:        
        for v in forcing:
            if 'LIFETIME_DEP_ant' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)

    elif 'tau_lifetime_dep_geo' in perturbation: 
        for v in forcing:
            if 'LIFETIME_DEP_geo' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)
                
    elif 'tau_lifetime_dep_chem' in perturbation:        
        for v in forcing:
            if 'LIFETIME_DEP_chem' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)

    elif 'tau_lifetime_dep_bb' in perturbation:        
        for v in forcing:
            if 'LIFETIME_DEP_bb' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)

    elif 'tau_lifetime_dep_soil' in perturbation:        
        for v in forcing:
            if 'LIFETIME_DEP_soil' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)                                

    elif 'tau_lifetime_dep_ocn' in perturbation:        
        for v in forcing:
            if 'LIFETIME_DEP_ocn' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_DEP','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_DEP','_CHEM')]) #lifetime estimate (yr-1)                

    elif 'tau_lifetime_chem_all' in perturbation:        
        for v in forcing:
            if 'LIFETIME_CHEM' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_CHEM','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_CHEM','_DEP')]) #lifetime estimate (yr-1)                

    elif 'tau_lifetime_chem_ocn' in perturbation:        
        for v in forcing:
            if 'LIFETIME_CHEM_ocn' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_CHEM','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_CHEM','_DEP')]) #lifetime estimate (yr-1)                

    elif 'tau_lifetime_chem_soil' in perturbation:        
        for v in forcing:
            if 'LIFETIME_CHEM_soil' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_CHEM','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_CHEM','_DEP')]) #lifetime estimate (yr-1)                

    elif 'tau_lifetime_chem_bb' in perturbation:        
        for v in forcing:
            if 'LIFETIME_CHEM_bb' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_CHEM','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_CHEM','_DEP')]) #lifetime estimate (yr-1)

    elif 'tau_lifetime_chem_ant' in perturbation:        
        for v in forcing:
            if 'LIFETIME_CHEM_ant' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_CHEM','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_CHEM','_DEP')]) #lifetime estimate (yr-1)                
                
    elif 'tau_lifetime_chem_chem' in perturbation:        
        for v in forcing:
            if 'LIFETIME_CHEM_chem' in v:
                forcing_pert[v] = forcing_pert[v]*scale
                forcing_pert[v.replace('_CHEM','')] = 1./(1/forcing_pert[v] + 1/forcing_pert[v.replace('_CHEM','_DEP')]) #lifetime estimate (yr-1)                
    elif 'source_all_no_ch4_no_geo' in perturbation:
        for s in forcing_pert:
            if 'SOURCE_' in s:
                if ('geo' in s) == False:
                    if ('chem' in s)==False:
                        forcing_pert[s] = forcing_pert[s]*scale
                    else:
                        forcing_pert['chem_nch4']   = forcing_pert['chem_nch4']*scale
                        forcing_pert['SOURCE_chem'] = forcing_pert['chem_ch4']+forcing_pert['chem_nch4']
    elif 'source_all_no_ch4' in perturbation:        
        for s in forcing_pert:
            if 'SOURCE_' in s:
                if ('chem' in s)==False:
                    forcing_pert[s] = forcing_pert[s]*scale
                else: 
                    forcing_pert['chem_nch4']   = forcing_pert['chem_nch4']*scale
                    forcing_pert['SOURCE_chem'] = forcing_pert['chem_ch4']+forcing_pert['chem_nch4']
    elif 'source_all_no_geo' in perturbation:
        for s in forcing_pert:
            if 'SOURCE_' in s:
                if ('geo' in s) == False:
                    forcing_pert[s] = forcing_pert[s]*scale                
    elif 'source_all' in perturbation:
        for s in forcing_pert:
            if 'SOURCE_' in s:
                forcing_pert[s] = forcing_pert[s]*scale
    elif 'emission_all' in perturbation:
        for s in forcing_pert:
            if 'SOURCE_' in s:
                if ('chem' in s) == False:
                    forcing_pert[s] = forcing_pert[s]*scale
        forcing_pert['SOURCE'] = 0.
        for s in forcing_pert:
            if 'SOURCE_' in s:
                forcing_pert['SOURCE'] = forcing_pert['SOURCE'] + forcing_pert[s]
    elif 'emission_no_geo' in perturbation:
        for s in forcing_pert:
            if 'SOURCE_' in s:
                if ('chem' in s) == False and ('geo' in s) == False:
                    forcing_pert[s] = forcing_pert[s]*scale
        forcing_pert['SOURCE'] = 0.
        for s in forcing_pert:
            if 'SOURCE_' in s:
                forcing_pert['SOURCE'] = forcing_pert['SOURCE'] + forcing_pert[s]
    elif 'source_anthro' in perturbation:
        forcing_pert['SOURCE_ant']  = forcing_pert['SOURCE_ant']*scale
    elif 'source_bb' in perturbation:
        forcing_pert['SOURCE_bb']   = forcing_pert['SOURCE_bb']*scale
    elif 'source_nat_bb' in perturbation:
        forcing_pert['SOURCE_ocn']   = forcing_pert['SOURCE_ocn']*scale
        forcing_pert['SOURCE_soil']  = forcing_pert['SOURCE_soil']*scale
        forcing_pert['SOURCE_bb']    = forcing_pert['SOURCE_bb']*scale
    elif 'source_nat' in perturbation:
        forcing_pert['SOURCE_ocn']   = forcing_pert['SOURCE_ocn']*scale
        forcing_pert['SOURCE_soil']  = forcing_pert['SOURCE_soil']*scale
    elif 'source_soil' in perturbation:
        forcing_pert['SOURCE_soil'] = forcing_pert['SOURCE_soil']*scale        
    elif 'source_ocn' in perturbation:
        forcing_pert['SOURCE_ocn']  = forcing_pert['SOURCE_ocn']*scale              
    elif 'source_chem_nch4' in perturbation:
        forcing_pert['chem_nch4'] = forcing_pert['chem_nch4']*scale
        forcing_pert['SOURCE_chem'] = forcing_pert['chem_ch4']+forcing_pert['chem_nch4']
    elif 'source_chem_ch4' in perturbation:
        forcing_pert['chem_ch4'] = forcing_pert['chem_ch4']*scale
        forcing_pert['SOURCE_chem'] = forcing_pert['chem_ch4']+forcing_pert['chem_nch4']
        forcing_pert['LIFETIME_CHEM'] = forcing_pert['LIFETIME_CHEM']/scale
        for vv in ['bb','chem','ant','soil','ocn','geo']:
            forcing_pert['LIFETIME_CHEM_'+vv] = forcing_pert['LIFETIME_CHEM_'+vv]/scale                                            
    elif 'source_chem' in perturbation:
        forcing_pert['SOURCE_chem'] = forcing_pert['SOURCE_chem']*scale              
    elif "transport_Fo_all" in perturbation:
        for v in forcing:
            if 'Fo' in v:
                forcing_pert[v] = forcing_pert[v] * scale
    elif "transport_Fo_ant" in perturbation:
        forcing_pert['Fo_ant'] = forcing_pert['Fo_ant'] * scale
    elif "transport_Fo_bb" in perturbation:
        forcing_pert['Fo_bb'] = forcing_pert['Fo_bb'] * scale
    elif "transport_Fo_chem" in perturbation:
        forcing_pert['Fo_chem'] = forcing_pert['Fo_chem'] * scale
    elif "transport_Fo_soil" in perturbation:
        forcing_pert['Fo_soil'] = forcing_pert['Fo_soil'] * scale
    elif "transport_Fo_ocn" in perturbation:
        forcing_pert['Fo_ocn'] = forcing_pert['Fo_ocn'] * scale
    elif "transport_A1_all" in perturbation:
        for v in forcing:
            if 'A1' in v:
                forcing_pert[v] = forcing_pert[v] * scale
    elif "transport_A2_all" in perturbation:
        for v in forcing:
            if 'A2' in v:
                forcing_pert['A2'] = forcing_pert['A2'] * scale
    else:
        print("perturbation not found")
        exit()
    
    return(forcing_pert)

def calc_jacobian_parallel(model_ref,forcing,it_start=0,ntimes=None,output_jacobian=None,observation=None,iter=0,n_jobs=4,n_jobs_time=1,optimize=[],inversion_resolution='monthly',**kwargs):

    if type(observation) is dict:
        observation = np.array([observation])

    if ntimes is None:
        ntimes      = len(forcing['time'])-it_start        

    if len(optimize)>1:

        if n_jobs==1:
            [calc_jacobian_parallel(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,
                                    optimize=[optimize[p]],output_jacobian=output_jacobian,inversion_resolution=inversion_resolution,iter=iter,observation=observation,n_jobs=1,n_jobs_time=n_jobs_time,**kwargs) for p in np.arange(len(optimize))]
        else:            
            Parallel(n_jobs=min([n_jobs,len(optimize)]),verbose=10)(delayed(calc_jacobian_parallel)(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,
                                                                                        optimize=[optimize[p]],output_jacobian=output_jacobian,
                                                                                        inversion_resolution=inversion_resolution,observation=observation,iter=iter,n_jobs=1,n_jobs_time=n_jobs_time,**kwargs)
                                                                    for p in np.arange(len(optimize)))

    else:

        target = []
        if type(observation) is dict:
            observation = np.array([observation])

        for io in range(len(observation)):
            if 'target' in observation[io]:
                target = target+[observation[io]['target']]
            else:
                target = target+['H2']

        jacob_file = output_jacobian+"/%s_resolution=%s_time_start=%u_ntimes=%u_%s_%u.npy"%(optimize[0]['name'],inversion_resolution,it_start,ntimes,"VARIABLE",iter)

        if os.path.isfile(jacob_file.replace("VARIABLE",target[0]))==False:
            if 'no_time' in optimize[0]['reference']:
                Jacob =calc_jacobian(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,optimize=[optimize[0]],output_jacobian=None,inversion_resolution=inversion_resolution,target=target,**kwargs)

            elif 'mclimo' in optimize[0]['reference']:
                print("n_jobs=%u n_jobs_time=%u"%(n_jobs,n_jobs_time))
                if np.max([n_jobs,n_jobs_time])==1:
                    Jacob =calc_jacobian(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,optimize=[optimize[0]],output_jacobian=None,inversion_resolution=inversion_resolution,target=target,**kwargs)                
                else:
                    results =Parallel(n_jobs=np.max([n_jobs,n_jobs_time]),verbose=10)(delayed(calc_jacobian)(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,it_start_p=its,ntimes_p=(its+1),optimize=[optimize[0]],output_jacobian=None,inversion_resolution=inversion_resolution,target=target,**kwargs) for its in np.arange(0,12))

                    Jacob = {}
                    for v in results[0]:
                        Jacob[v]   = 0.
                        for n in range(len(results)):
                            Jacob[v] = results[n][v]+Jacob[v]

            else:
                print("n_jobs=%u n_jobs_time=%u"%(n_jobs,n_jobs_time))
                if np.max([n_jobs,n_jobs_time])==1:
                    results =[calc_jacobian(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,it_start_p=its,ntimes_p=12,optimize=[optimize[0]],output_jacobian=None,inversion_resolution=inversion_resolution,target=target,**kwargs) for its in np.arange(it_start,it_start+ntimes,12)]
                else:
                    results =Parallel(n_jobs=np.max([n_jobs,n_jobs_time]),verbose=10)(delayed(calc_jacobian)(model_ref=model_ref,forcing=forcing,it_start=it_start,ntimes=ntimes,it_start_p=its,ntimes_p=12,
                                                                                   optimize=[optimize[0]],output_jacobian=None,inversion_resolution=inversion_resolution,target=target,**kwargs) for its in np.arange(it_start,it_start+ntimes,12))


                Jacob = {}
                for v in results[0]:
                    Jacob[v]   = 0.

                    for n in range(len(results)):
                        Jacob[v] = results[n][v]+Jacob[v]

            print(Jacob)
            for v in Jacob:
                np.save(jacob_file.replace("VARIABLE",v),Jacob[v])

def calc_jacobian(model_ref,forcing,it_start=0,ntimes=None,optimize=[],output_jacobian=None,inversion_resolution='monthly',it_start_p=None,ntimes_p=None,
                  transport='monthly',scale_pert=1.10,tag=False,iter=0,target=['H2'],compound="H2",col_order=None,**kwargs):


    # with open(os.path.join('tmp','debug.pkl'), 'wb') as f:
    #     pickle.dump({'model_ref':model_ref,'forcing':forcing,'it_start':it_start,'ntimes':ntimes,
    #                  'optimize':optimize,'output_jacobian':output_jacobian,'inversion_resolution':inversion_resolution,
    #                  'it_start_p':it_start_p,'ntimes_p':ntimes_p,'transport':transport,
    #                  'scale_pert':scale_pert,'tag':tag,'iter':iter,'target':target,'compound':compound,'col_order':col_order,**kwargs}, f)

    
    if ntimes is None:
        ntimes      = len(forcing['time'])-it_start

    for s in special_regions:
        special_columns[s] = []
        ibt=0        
        for ct in col_order:
            if ct in special_regions[s]:
                special_columns[s] = special_columns[s]+[ibt]
            ibt+=1

        special_columns[s] = np.array(special_columns[s])

    nbox_opt = {}
    for v in model_ref:
        if ('_half' in v)==False and ("source" in v)==False:
            nbox_opt[v]        = len(model_ref[v][0,:])
    pert     = optimize

    M0 = {}
    for v in model_ref:
        M0[v] = model_ref[v][it_start,:]
    if tag==False:
        M0    = M0[compound+'_half']    
        
    Jacob = {}
    m_ref = {}        

    if inversion_resolution=="annual":
        dt = 12
        for v in target:
            m_ref[v] = calc_annual(model_ref[v],forcing,target,**kwargs)[int(it_start/dt):int((it_start+ntimes)/dt)]
    else:
        dt = 1            
        for v in target:
            m_ref[v] = model_ref[v][it_start:(it_start+ntimes),:]
   
    print(pert)
    for p in range(len(pert)):
        
        J = pert[p]

        if (output_jacobian is None)==False:
            jacob_file = output_jacobian+"/%s_resolution=%s_time_start=%u_ntimes=%u_%s_%u.npy"%(J['name'],inversion_resolution,it_start,ntimes,"[VARIABLE]",iter)

        ntimes_j = int(ntimes/dt)

        print("J_ref",J['reference'])

        if (output_jacobian is None) or (os.path.isfile(jacob_file.replace("[VARIABLE]","H2"))==False): 
            
            nbox                 = len(J['box'])
            print("Calculate jacobian for %s..."%J['reference'])
            Jacob = {}

            if 'no_time' in J['reference']:
                for v in target:
                    Jacob[v]   = zeros([ntimes_j*nbox_opt[v],1,nbox])                

                count     = 0                                        
                for ib in J['box']:
                    
                    if "transport_Fo" in J['reference']:
                        scale = np.ones(len(forcing['Fo']))
                    elif "transport_A1" in J['reference']:
                        scale = np.ones(len(forcing['A1']))
                    elif "transport_A2" in J['reference']:
                        scale = np.ones(len(forcing['A2']))
                    else:
                        scale = np.ones_like(forcing['LIFETIME_DEP'][0,:])

                    is_special = False
                    for s in special_regions:
                        if ib==s:
                            scale[special_columns[s]] = scale_pert
                            is_special=True
                    if (is_special==False):
                        scale[ib] = scale_pert                    


                    forcing_pert = perturb_forcing(forcing,J['reference'],scale)

                    model_pert   = run_model(forcing_pert,transport,
                                             it_start=it_start,
                                             ntimes=ntimes,
                                             tag=tag,col_order=col_order,
                                             M0=M0,
                                             compound=compound,**kwargs)
                    
                    if (it_start>0):
                        for v in model_pert:
                            model_pert[v][0:it_start,:] = model_ref[v][0:it_start,:]

                    for v in target:
                        if inversion_resolution=="annual":
                            m_pert = calc_annual(model_pert[v],forcing,v,**kwargs)[int(it_start/dt):int((it_start+ntimes)/dt)]
                        else:
                            m_pert = model_pert[v][it_start:(it_start+ntimes)]

                        delta_mod = (m_pert-m_ref[v])/(scale_pert-1.)      
                        delta_mod = delta_mod[:,0:nbox_opt[v]]

                        Jacob[v][:,0,count] = np.reshape(delta_mod,ntimes_j*nbox_opt[v],order='F')
                            
                    count+=1

            elif 'mclimo' in J['reference']:

                print("mclimo - calc_jacobian")
                for v in target:
                    Jacob[v]  = zeros([ntimes_j*nbox_opt[v],12,nbox])                
            
                #this is an ugly way to allow parallelization
                if (it_start_p is None):
                    it_start_p     = 0
                if (ntimes_p is None):
                    ntimes_p = 12                               

                for it in range(it_start_p,ntimes_p,1):

                    print("%2.1f%s"%(100*it/12,"%"))

                    count     = 0                    
                    for ib in J['box']:

                        scale = np.ones_like(forcing['LIFETIME_DEP'])

                        is_special = False
                        for s in special_regions:
                            if ib==s:
                                is_special = True
                                for ibt in special_columns[s]:
                                    scale[it::12,ibt]  = scale_pert
                        if (is_special==False):
                            scale[it::12,ib] = scale_pert

                        forcing_pert = perturb_forcing(forcing,J['reference'],scale)

                        model_pert = run_model(forcing_pert,transport,it_start=it_start,
                                               ntimes=ntimes,
                                               tag=tag,col_order=col_order,
                                               M0=M0,
                                               compound=compound,**kwargs)

                        if (it_start>0):
                            for v in model_pert:
                                model_pert[v][0:it_start,:] = model_ref[v][0:it_start,:]


                        m_pert = {}
                        for v in target:
                            if inversion_resolution=="annual":
                                m_pert = calc_annual(model_pert[v],forcing,v,**kwargs)[int(it_start/dt):int((it_start+ntimes)/dt)]
                            else:
                                m_pert = model_pert[v][it_start:(it_start+ntimes)]

                            delta_mod = (m_pert-m_ref[v])/(scale_pert-1.)      
                            delta_mod = delta_mod[:,0:nbox_opt[v]]

                            Jacob[v][:,it,count] = np.reshape(delta_mod,ntimes_j*nbox_opt[v],order='F')
                        count+=1                    
            else:
                for v in target:
                    Jacob[v]  = zeros([ntimes_j*nbox_opt[v],ntimes_j,nbox])                
            
                #this is an ugly way to allow parallelization
                if (it_start_p is None):
                    it_start_p     = it_start
                if (ntimes_p is None):
                    ntimes_p = ntimes
                
                for it in range(it_start_p,it_start_p+ntimes_p,dt):

                    print("%2.1f%s"%(100*(it-it_start)/ntimes,"%"))

                    count     = 0                    
                    for ib in J['box']:

                        scale = np.ones_like(forcing['LIFETIME_DEP'])

                        is_special = False
                        for s in special_regions:
                            if ib==s:
                                is_special = True
                                for ibt in special_columns[s]:
                                    scale[np.arange(it,it+dt),ibt]  = scale_pert
                        if (is_special==False):
                            scale[np.arange(it,it+dt),ib] = scale_pert

                        forcing_pert = perturb_forcing(forcing,J['reference'],scale)

                        model_pert = run_model(forcing_pert,transport,it_start=it_start,
                                               ntimes=ntimes,
                                               tag=tag,col_order=col_order,
                                               M0=M0,
                                               compound=compound,**kwargs)

                        if (it_start>0):
                            for v in model_pert:
                                model_pert[v][0:it_start,:] = model_ref[v][0:it_start,:]


                        m_pert = {}
                        for v in target:
                            if inversion_resolution=="annual":
                                m_pert = calc_annual(model_pert[v],forcing,v,**kwargs)[int(it_start/dt):int((it_start+ntimes)/dt)]
                            else:
                                m_pert = model_pert[v][it_start:(it_start+ntimes)]

                            delta_mod = (m_pert-m_ref[v])/(scale_pert-1.)      
                            delta_mod = delta_mod[:,0:nbox_opt[v]]

                            Jacob[v][:,int((it-it_start)/dt),count] = np.reshape(delta_mod,ntimes_j*nbox_opt[v],order='F')
#                            print(v,mean(Jacob[v]),min(Jacob[v]),max(Jacob[v]))
                        count+=1

            if (output_jacobian is None)==False:
                for v in Jacob:
                    print("SAVING ",jacob_file.replace("[VARIABLE]",v))
                    np.save(jacob_file.replace("[VARIABLE]",v),Jacob)
            else:
                #for debugging                
                np.save("tmp/tmp.npy",Jacob)
                print("returning to calling routine")
                return(Jacob)

            print("...Done")
                
def calc_optimize(model_ref,xi,forcing,forcing_p,output_jacobian,output_optimize,observation,optimize,it_start=0,ntimes=None,inversion_resolution='monthly',col_order=None,iter=0,gamma=0,n_jobs=4,n_jobs_time=1,compound="H2",**kwargs):

    F  = np.array([])
    y  = np.array([])
    So = np.array([])
    K  = np.array([])

    #THIS IS NEEDED TO PERFORM MATRIX OPERATIONS IN PARALLEL. https://superfastpython.com/numpy-number-blas-threads/
    nj = np.max([n_jobs,n_jobs_time])
    environ['OMP_NUM_THREADS']        = "%u"%nj
    environ['OPENBLAS_NUM_THREADS']   = "%u"%nj
    environ['MKL_NUM_THREADS']        = "%u"%nj
    environ['VECLIB_MAXIMUM_THREADS'] = "%u"%nj
    environ['NUMEXPR_NUM_THREADS']    = "%u"%nj
    
    if ntimes is None:
        ntimes      = len(forcing['time'])-it_start

    #for backward compatibility
    if type(observation) is dict:
        observation = np.array([observation])
    
    nbox_opt = np.zeros(len(observation)).astype(int)

    for io in range(len(observation)):
        if ('target' in observation[io])==False:
            target = "H2"
        else:
            target = observation[io]['target']

        h2_avg = model_ref[target]

        if inversion_resolution=='annual':
            dt=12
            m_ref = calc_annual(h2_avg,forcing,target,**kwargs)[int(it_start/dt):int((it_start+ntimes)/dt)]
            time  = pd.to_datetime(forcing['time'][it_start:(it_start+ntimes):12].year,format='%Y')
        else:
            dt=1        
            m_ref = h2_avg[it_start:(it_start+ntimes),:]
            time  = forcing['time'][it_start:(it_start+ntimes)]
        
        ntimes_j = int(ntimes/dt)

        nbox_opt[io]        = len(m_ref[0,:])

        FF  = np.reshape(m_ref,ntimes_j*nbox_opt[io],order='F')
        F   = np.concatenate([F,FF])

        #read observations
        obs = pd.read_csv(observation[io]['file'],parse_dates=[0]).set_index('time')

        for c in col_order:
            if (c in obs.columns)==False:
                obs[c] = np.nan

        obs = obs[col_order]

        #obs needs to be mapped onto model time. This should be made more robust.
        obs = obs.reindex(obs.index.union(time))
        obs = obs.loc[time] #important when observations are longer than model

        observation[io]['processed'] = obs.to_numpy()

        #    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        #        print(obs)
        #    print(shape(obs))

        #obs needs to have the same number of boxes as the model. This should be made more robust
        y = np.concatenate([y,np.reshape(obs.to_numpy(),ntimes_j*nbox_opt[io],order='F')])

        first = True            
        for o in optimize:
            jacob_file = output_jacobian+"/%s_resolution=%s_time_start=%u_ntimes=%u_%s_%u.npy"%(o['name'],inversion_resolution,it_start,ntimes,target,iter)

            if 'no_time' in o ['reference']:
                Jacob      = np.reshape(np.load(jacob_file),[ntimes_j*nbox_opt[io],len(o['box'])],order='F')
                Sa_tmp     = np.diag(o['error']**2 * np.ones(len(o['box'])))
                #here we scale the error by ntimes. Otherwise, Sa_tmp is too large
                Sa_tmp     = Sa_tmp/ntimes_j
            elif 'mclimo' in o ['reference']:
                Jacob      = np.reshape(np.load(jacob_file),[ntimes_j*nbox_opt[io],12*len(o['box'])],order='F')
                Sa_tmp     = np.diag(o['error']**2 * np.ones(12*len(o['box'])))
                #here we scale the error by ntimes. Otherwise, Sa_tmp is too large
                Sa_tmp     = Sa_tmp/(ntimes_j/12)
            else:
                Sa_tmp     = np.diag(o['error']**2 * np.ones(ntimes_j*len(o['box'])))
                Jacob      = np.reshape(np.load(jacob_file),[ntimes_j*nbox_opt[io],ntimes_j*len(o['box'])],order='F')
                Sa_tmp     = add_temporal_correlation(Sa_tmp,ntimes=ntimes_j,inversion_resolution=inversion_resolution,**o)

            if first:
                Sa = copy.deepcopy(Sa_tmp)
                KK  = copy.deepcopy(Jacob)
                first = False
            else:
                Sa = block_diag(Sa,Sa_tmp)
                KK  = np.concatenate([KK,Jacob],axis=1)


        if len(K)==0:
            K = copy.deepcopy(KK)
        else:
            K = np.concatenate([K,KK])

        So_tmp = 0.
        #error covariance matrix
        if 'error' in observation[io]:
            if isinstance(observation[io]['error'],str): 
                error_h2 = pd.read_csv(observation[io]['error'],parse_dates=[0]).set_index('time')
                for c in col_order:
                    if (c in error_h2.columns)==False:
                        error_h2[c] = np.nan
                error_h2 = error_h2[col_order]                
                error_h2 = error_h2.reindex(error_h2.index.union(time))
                error_h2 = error_h2.loc[time]
                error_h2 = error_h2.to_numpy()                
            else:
                error_h2 = observation[io]['error']

            if np.isscalar(error_h2):
                So_tmp         = error_h2**2*np.ones([ntimes_j,nbox_opt[io]])
            else:
                if len(np.shape(error_h2))==1:
                    So_tmp = error_h2**2*np.ones([ntimes_j,1])
                else:
                    So_tmp = error_h2**2

        if 'error_rel' in observation[io]:
            So_tmp = So_tmp + (observation[io]['error_rel']*observation[io]['processed'])**2


        So = np.concatenate([So,np.reshape(So_tmp,ntimes_j*nbox_opt[io],order='F')])

    ind_valid = np.ravel(np.argwhere(np.isnan(y)==False))

    os.makedirs(output_optimize,exist_ok=True)

    np.save(os.path.join(output_optimize,"So.npy"),So)
    np.save(os.path.join(output_optimize,"F_%u.npy"%(iter-1)),F)
    np.save(os.path.join(output_optimize,"y.npy"),y)
    np.save(os.path.join(output_optimize,"Sa.npy"),Sa)
    
    y = y[ind_valid,None]
    F = F[ind_valid,None]
    K = K[ind_valid,:]

    print(np.sum(y))
    print(np.sum(F))

    yl = len(y)
    
#    So = np.diag(So[ind_valid])
    
    #Get inverse
    SaI  = inv(Sa)
#    SoI  = inv(So)
    SoI  = np.diag(1/So[ind_valid]) #This assumes observations don't have cross-correlations. 
    print("done calculating inverse")

    #NEED TO REVISE THIS FOR NON-LINEAR INVERSIONS
    xa = 1. #we should make this an input parameter but for now, keep it simple.
    #inversion Rodgers 5.36
#    print("SaI",np.shape(SaI))
#    print("SoI",np.shape(SoI))
    print("K",np.shape(K))
    print("SoI",np.shape(SoI))
    print("SaI",np.shape(SaI))
#    print("K.T@SoI@K",np.shape(K.T@SoI@K ))
    LHS  = SaI + K.T@SoI@K + gamma*SaI
    RHS  = K.T@SoI@(y-F)
    if hasattr(xi,"__len__"):
        RHS = RHS - SaI@(xi-xa)

    print("solving for dx")
    dx   = solve(LHS,RHS)
    print("done...")
    xhat = xi * (1 +  dx)

    np.save(os.path.join(output_optimize,"xhat_%u.npy"%iter),xhat)
    
    #posterior covariance matrix
    S_hat  = inv(K.T @ SoI @ K + SaI)
    #averaging kernel
#    AK_hat = np.eye(np.shape(Sa)[0]) - S_hat @ SaI
#    C_hat  = np.sqrt(np.diag(S_hat))
#    C_hat  = np.diag(1/C_hat)
#    C_hat  = C_hat @ S_hat @ C_hat
    
    
    #update forcing
    index = 0
    forcing_i = copy.deepcopy(forcing)

    output_scale = {}
    for o in optimize:

        if 'no_time' in o['reference']:            
            scale_opt = np.reshape(xhat[index:(index+len(o['box']))],len(o['box']),order='F')
            index     = index + len(o['box'])                        
        elif 'mclimo' in o['reference']:
            scale_opt = np.reshape(xhat[index:(index+12*len(o['box']))],[12,len(o['box'])],order='F')
            index     = index + 12*len(o['box'])                        
        else:            
            scale_opt = np.reshape(xhat[index:(index+ntimes_j*len(o['box']))],[ntimes_j,len(o['box'])],order='F')
            index     = index + ntimes_j*len(o['box'])

        print(o,"mean %f min %f max %f"%(mean(scale_opt),min(scale_opt),max(scale_opt)))
            
        #this is not applicable to transport
        if ("transport" in o['reference'])==False:
            if 'no_time' in o['reference']:
                if len(o['box'])==1 and (o['box'][0] in special_regions):
                    S = pd.DataFrame(data=scale_opt[None,:],columns=o['box'],index=time[[0]])
                else:
                    o['box'] = np.array(o['box']).astype(int)
                    S = pd.DataFrame(data=scale_opt[None,:],columns=col_order[o['box']],index=time[[0]])
            elif 'mclimo' in o['reference']:
                if len(o['box'])==1 and (o['box'][0] in special_regions):
                    S = pd.DataFrame(data=scale_opt,columns=o['box'],index=time[0:12])
                else:
                    o['box'] = np.array(o['box']).astype(int)
                    S = pd.DataFrame(data=scale_opt,columns=col_order[o['box']],index=time[0:12])

                print(S)
            else:
                if len(o['box'])==1 and (o['box'][0] in special_regions):
                    S = pd.DataFrame(data=scale_opt,columns=o['box'],index=time)
                else:
                    o['box'] = np.array(o['box']).astype(int)
                    S = pd.DataFrame(data=scale_opt,columns=col_order[o['box']],index=time)

            output_scale[o['name']] = convert_scale(S,time=forcing_i['time'][it_start:(it_start+ntimes)],col_order=col_order,reference=o['name'])

            output_scale[o['name']] = output_scale[o['name']].reindex(output_scale[o['name']].index.union(forcing_i['time']))

            output_scale[o['name']] = output_scale[o['name']].fillna(1.)
            scale_opt = output_scale[o['name']].to_numpy()
        else:
            scale_opt_tmp = copy.deepcopy(scale_opt)            
            scale_opt = np.ones(len(forcing[o['reference'].split('_')[1]]))
            scale_opt[o['box']] = scale_opt_tmp
            output_scale[o['name']] = pd.DataFrame(data=scale_opt[None,:],
                                                   columns=np.arange(len(scale_opt)),
                                                   index=time[[0]])

        forcing_i = perturb_forcing(forcing_i,o['reference'],scale_opt)

    #calculate chi2
    chi2_obs     = np.squeeze((y-F).T@SoI@(y-F))
    chi2_model   = 0
    if hasattr(xi,"__len__"):        
        chi2_model = (xi-xa).T@SaI@(xi-xa)
    chi2         = chi2_obs + chi2_model
                   
    #run model with new forcing    
    model_new = run_model(forcing_i,col_order=col_order,
                          **kwargs)

    F = np.array([])

    for io in range(len(observation)):
        if ('target' in observation[io])==False:
            target = "H2"
            observation[io]['target'] = "H2"
        else:
            target = observation[io]['target']

        h2_avg = model_new[target]

        if inversion_resolution=='annual':
            dt=12
            m_new = calc_annual(h2_avg,forcing,target,**kwargs)[int(it_start/dt):int((it_start+ntimes)/dt)]
            time  = pd.to_datetime(forcing['time'][it_start:(it_start+ntimes):12].year,format='%Y')
        else:
            dt=1        
            m_new = h2_avg[it_start:(it_start+ntimes),:]
            time  = forcing['time'][it_start:(it_start+ntimes)]        

        FF       = np.reshape(m_new,ntimes_j*nbox_opt[io],order='F')
        F        = np.concatenate([F,FF])        

    np.save(os.path.join(output_optimize,"F_%u.npy"%iter),F)        
    F        = F[ind_valid,None]

#    print("y",y[int((yl-1)/2)],y[int(yl-1)])
#    print("F",F[int((yl-1)/2)],F[int(yl-1)])      

    chi2_new_obs   = np.squeeze((y-F).T@SoI@(y-F)    )
    chi2_new_model = np.squeeze((xhat-xa).T@SaI@(xhat-xa))
    chi2_new       = chi2_new_obs + chi2_new_model


    print("chi2=%g chi2_model=%g chi2_obs=%g"%(chi2,chi2_model,chi2_obs))
    print("chi2_new=%g chi2_new_model=%g chi2_new_obs=%g"%(chi2_new,chi2_new_model,chi2_new_obs))    
    
#    np.save(os.path.join(output_optimize,"AK_hat_%u.npy"%iter),AK_hat)
#    np.save(os.path.join(output_optimize,"C_hat_%u.npy"%iter),C_hat)
    if (gamma>0) and (chi2_new>chi2):
        #increase gamma and reject this iteration
        print(" <xxxxxxxxxxxxxxxxxxxxxxxxxxxx>")
        print(" <xxxx  reject iteration  xxxx>")
        print(" <xxxxxxxxxxxxxxxxxxxxxxxxxxxx>")        
        gamma     = gamma*2
        xhat      = xi
        forcing_i = forcing_p

        #no need to recalculate jacobian
        print("linking jacobian")
        for o in optimize:
            all_jacob_file = output_jacobian+"/%s_resolution=%s_time_start=%u_ntimes=%u_%s_%u.npy"%(o['name'],inversion_resolution,it_start,ntimes,"*",iter)            
            for jacob_file in glob.glob(all_jacob_file):
                jacob_file_new = jacob_file.replace("_%u.npy"%iter,"_%u.npy"%(iter+1))
                os.system("ln -sf "+jacob_file.split('/')[-1]+" "+jacob_file_new)
            
    else:        
        #decrease gamma for next step
        print(" >============================<")        
        print(" ====>  accept iteration  <====")
        print(" >============================<")                
        gamma     = gamma/2
        chi2      = chi2_new

        print("Start saving")

        for o in output_scale:
            save_file(output_scale[o],os.path.join(output_optimize,o+'_%u.adj'%iter))
            save_file(output_scale[o],os.path.join(output_optimize,o+'_best.adj'))
#THIS CAN BE VERY SLOW. NOT SURE WHY. 
#        Parallel(n_jobs=nj,verbose=10)(delayed(save_file)(output_scale[o],os.path.join(output_optimize,o+'_%u.csv'%iter)) for o in output_scale)
#        Parallel(n_jobs=nj,verbose=10)(delayed(save_file)(output_scale[o],os.path.join(output_optimize,o+'_best.csv')) for o in output_scale)                                                            

        print("Done with saving")
                                           
        for o in output_scale:
            print(o)
            output_scale[o].plot()
            plt.savefig(os.path.join(output_optimize,o+'_%u.pdf'%iter))
            plt.savefig(os.path.join(output_optimize,o+'_best.pdf'))
            np.save(os.path.join(output_optimize,"S_hat_best.npy"),S_hat)

            #convenient to save in FORCING structure
            #only do this for H2
            if compound=="H2":
                for v in ['bb','chem','ant','soil','ocn','geo']:
                    forcing_i['SOIL_SINK_%s'%v] = model_new['H2_sink_dep_%s'%v]
                    forcing_i['CHEM_SINK_%s'%v] = model_new['H2_sink_chem_%s'%v]
                    forcing_i['BURDEN_%s'%v]    = model_new['H2_burden_%s'%v]                
                forcing_i['SOIL_SINK'] = model_new['H2_sink_dep']
                forcing_i['CHEM_SINK'] = model_new['H2_sink_chem']
                forcing_i['BURDEN']    = model_new['H2_burden']                
            
            with open(os.path.join(output_optimize,'forcing_best.pkl'), 'wb') as f:
                pickle.dump(forcing_i, f)

            plt.close()


    print("returning to calling routine")
    return(xhat,forcing_i,gamma)
                        
def main(argv):

    opts,args = getopt.getopt(argv,"r:o:dh",["run=","optimize=","debug","help"])

    run_param  = None
    opt_param  = {}
    debug      = False

    for opt,arg in opts:
        if opt in ("-h","--help"):
            print("python box_model.py -r [run] -o [optimize] -d [debug]")
            exit()
        if opt in ("-d","--debug"):            
            debug = True
        if opt in ("-r","--run"):
            with open(arg, 'r') as f:
                run_param = json.load(f)            
        if opt in ("-o","--optimize"):
            with open(arg, 'r') as f:
                opt_param = json.load(f)

    if ('trop' in run_param)==False:
        run_param['trop']=True

    if run_param['trop']:
        run_param['col_order'] = gcol_order[0:6]
    else:
        run_param['col_order'] = gcol_order

    FORCING,GLOBAL_MODEL_INPUT = read_forcing(**run_param,**opt_param)
    
    out_ppb = run_model(FORCING,**run_param)

    if ("output_dir" in run_param)==False:
        run_param['output_dir']="output"

    run_param['output_dir'] = os.path.join(run_param['output_dir'],run_param['name'])    
    os.makedirs(run_param['output_dir'],exist_ok=True)
    save_output(out_ppb,FORCING,**run_param,**opt_param)

    FORCING_i = copy.deepcopy(FORCING)
    scale_i   = 1.

    if debug:
        diagnostics_figure(GLOBAL_MODEL_INPUT,out_ppb,FORCING,**run_param)

    if len(opt_param)>0:

        if "iter_max" in opt_param:
            iter_max = opt_param['iter_max']
        else:
            iter_max = 1

        iter    = 0
        proceed = True
        out_ppb = run_model(FORCING_i,**run_param)
        out_ppb_ref = copy.deepcopy(out_ppb)

        while (proceed):

            output_jacobian = os.path.join(run_param['output_dir'],opt_param['opt_name'],"jacobian")            
            os.makedirs(output_jacobian,exist_ok=True)
            calc_jacobian_parallel(model_ref=out_ppb,forcing=FORCING_i,output_jacobian=output_jacobian,
                                   iter=iter,
                                   **run_param,**opt_param)
            output_optimize = os.path.join(run_param['output_dir'],opt_param['opt_name'],"optimize")

            print("*************************")
            print("*     ITERATION %03u     *"%iter)
            print("*************************")

            
            scale_i_new,FORCING_i,opt_param['gamma']                = calc_optimize(model_ref=out_ppb,
                                                                      xi=scale_i,
                                                                      forcing=FORCING,
                                                                      forcing_p=FORCING_i,
                                                                      output_jacobian=output_jacobian,
                                                                      output_optimize=output_optimize,
                                                                      iter=iter,
                                                                      **run_param,**opt_param)  
            #not sure that's the best way            
            abs_change = abs(scale_i_new-scale_i)            
            rel_change = abs_change/scale_i
            
            abs_change = np.max(abs_change)
            rel_change = np.max(rel_change)

            if (iter>=iter_max):
                print("iter>iter_max -> stopping")
                proceed = False
            elif (opt_param['gamma']==0):
                print("no iteration -> stopping")
            elif (rel_change < rel_change_tol) and (abs_change < abs_change_tol) and (abs_change>0):
                print("tolerances met -> stopping")                
                proceed = False
            
            scale_i = scale_i_new
                
            print("abs. change=%f rel. change=%f gamma=%f"%(abs_change,rel_change,opt_param['gamma']))
            print("************************")            

            out_ppb = run_model(FORCING_i,**run_param)
            iter+=1

        save_output(out_ppb,FORCING,save_dir=output_optimize,**run_param,**opt_param)
        optimize_figure(model_ref=out_ppb_ref,model_opt=out_ppb,output_optimize=output_optimize,forcing=FORCING,**run_param,**opt_param)

def add_temporal_correlation(Si,ntimes,tau=0,inversion_resolution='monthly',**kwargs):
    
    S = copy.deepcopy(Si)    
    nS = np.shape(S)[0]
    
    if tau>0:
        if inversion_resolution=='monthly':
            tau = tau*12

        print("tau=",tau)

        for nb in range(int(nS/ntimes)):
            start = nb*ntimes
            for k in range(start,start+ntimes):
                for kk in range(start,start+ntimes): 
                    S[k,kk] = np.sqrt(S[k,k])*np.sqrt(S[kk,kk]) * np.exp(-np.abs(k-kk)/tau)
        
    return(S)

def save_file(data,output_file):
    data.to_csv(output_file)
        
if __name__ ==  '__main__':
      main(sys.argv[1:])
