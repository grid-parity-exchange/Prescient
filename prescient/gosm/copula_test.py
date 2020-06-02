#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

from timer import Timer,tic,toc
from copula import GaussianCopula, FrankCopula,StudentCopula,GumbelCopula,ClaytonCopula
import scipy.integrate as spi
from base_distribution import BaseDistribution
from base_distribution import MultiDistr
import numpy as np
import scipy.stats as sps
import scipy.special as sps
import matplotlib.pyplot as plt
from distributions import MultiNormalDistribution, UnivariateNormalDistribution,UnivariateStudentDistribution
from distributions import UnivariateEpiSplineDistribution, UnivariateEmpiricalDistribution
from vine import CVineCopula,DVineCopula
from mpl_toolkits.mplot3d import Axes3D
from copula_experiments.copula_diagonal import diag
from copula_experiments.copula_evaluate import RankHistogram,emd_sort
import datetime as dt
import gosm_options
from uncertainty_sources import Source,DataSources
from distribution_factory import distribution_factory
import os
import pandas as pd
import math
import segmenter
from scipy.stats import kendalltau, pearsonr, spearmanr

#Default value :
n=1000
datatype='actuals'
kind='diagonal'
source='wind'
dps=[10,14]
index=0
marginal_string = 'univariate-epispline'
bounds_list = [(0,1),(0,0.1),(0.9,1)]
copula_list = ['gaussian-copula', 'student-copula',
               'frank-copula', 'gumbel-copula', 'clayton-copula','independence-copula']
univariate_list = ['univariate-uniform','univariate-normal', 'univariate-epispline',
                   'univariate-empirical', 'univariate-student']
method = 'daytoday'
marginal_data ='hour' #same_hour
segment_marginal = ''
data_file = 'copula_experiments/datas_BPA_all.csv'
dimkeys = ['EH1','EH2']
path = './copula_experiments/BPA/' + source + '_' + datatype + '_' + str(dimkeys) + '_' + method



def creates_data():
    """
    This file clean the datas and put them in the shape you want to use them in the tests below.
    :return:
    """
    df_forecast = pd.DataFrame.from_csv('./copula_experiments/BPA/result_pacific.csv')
    df_actuals = pd.DataFrame.from_csv('./copula_experiments/BPA/actual_hourly.csv')
    df = pd.DataFrame(None,index=df_forecast.index,columns=['EH1','EH2','FH1','FH2','actuals','normal-sample','uniform-sample'])
    for i in df_forecast.index:
        try:
            df['EH1'][i] = df_forecast['Hr01'][i]-df_actuals['wind_actual'][i+dt.timedelta(hours=1)]
            df['EH2'][i] = df_forecast['Hr02'][i] - df_actuals['wind_actual'][i+dt.timedelta(hours=2)]
            df['FH1'][i] = df_forecast['Hr01'][i]
            df['FH2'][i] = df_forecast['Hr02'][i]
            df['actuals'][i] = df_actuals['wind_actual'][i]
        except:
            print(i)
            print(len(df_forecast.index))
            print(len(df.index))
            df.drop([i],inplace=True)
            print(len(df.index))
    l = len(df.index)
    df['normal-sample'] = np.random.randn(l)
    df['uniform-sample'] = np.random.rand(l)
    df2 = df.convert_objects(convert_numeric=True)
    df3 = df.dropna(axis=0, how='any')
    df3.to_csv('./copula_experiments/datas_BPA_all.csv')

def df_to_hourdf(data_file):
    df = pd.DataFrame.from_csv(data_file)
    l = len(df.index)
    list_dates = pd.date_range(df.index[0], end=df.index[l - 1]).date
    df_FH = pd.DataFrame.from_csv('./copula_experiments/datas_BPA_FH1_hours.csv')
    column = []
    for i in range(0,24):
        column.append(str(i))
    for i in range(0,24):
        column.append('FH'+str(i))

    for j in df.columns:
        print(j)
        df_pivoted = pd.DataFrame(index=list_dates,columns=column)
        for h in range(0,24):
            print(h)
            dfathour = df.loc[df.index.hour == h,j]
            for d in list_dates:
                try:
                    df_pivoted[str(h)][d]= dfathour.loc[dfathour.index.date == d][0]
                    df_pivoted['FH'+str(h)][d]=df_FH[str(h)][d]
                except:
                    pass

        df_pivoted.to_csv(data_file +'_'+str(j)+'_hours.csv')

def create_S(n=n,datatype=datatype,kind=kind,source=source,dps=dps,index=index,
             copula_list=copula_list,marginal_string=marginal_string,method=method,marginal_data=marginal_data,segment_marginal = None):
    """
    This function creates the S vector (cf diagonal_projection.pdf) and the histograms of S
    :param n: number of sample you want for U
    :param datatype:
    :param kind: type of projection : "diagonal", "kendall" or "marginal"
    :param source: "wind" or "solar" (only used to give names to files and folder)
    :param dps:
    :param index: index of the diagonal/marginal where you want to project
    :param copula_list: List of copula you want to try to fit.
    :param marginal_string: which marginal do you want to use ex : "univariate-empirical"
    :param method:
    :param marginal_data:
    :param segment_marginal: "segmented" or None you decide the segmentation parameters in the option files
    :return: creates the S csv file and the histograms png files
    """
    path = './copula_experiments/BPA/' + source + '_' + datatype + '_' + str(dimkeys)+'_'+method
    if not os.path.exists(path):
        os.makedirs(path)
    df_data = pd.DataFrame.from_csv('copula_experiments/datas_BPA_'+datatype+'_hours.csv')
    subset =[]
    for i in dimkeys:
        subset.append(i)
        subset.append('FH'+i)
    df = df_data.dropna(axis=0, how='any', subset=subset)
    l = len(df.index)
    if method == 'wholeyear':
        dt_list = df.index
    elif method == 'daytoday':
        dt_list = df.index[100:l - 1]

    df_S = pd.DataFrame(None, index=dt_list, columns=copula_list)
    for copula_string in copula_list:
        print('Computing '+source+' '+datatype+' '+kind+str(index)+' '+copula_string+' '+marginal_string+' '+method+' '+marginal_data)
        S = create_files(n, copula_string, path, df, dimkeys, index, kind, marginal_string,method,segment_marginal)

        df_S.loc[:, copula_string] = S


    suffix= return_suffix(kind,index,dps,marginal_string=marginal_string,marginal_data=marginal_data,segment_marginal=segment_marginal)

    df_S.to_csv(path + '/S_' + suffix)

    csv_to_histogram(path + '/S_' + suffix)
    print(path + '/S_' + suffix+' created')

def csv_to_histogram(path=None,n=10,columns =None, fit =None):
    """

    :param path: path where is your csv file that represent a datafram df
    :param n: Number of bins of your histogram
    :param columns(list of str): the columns of your dataframe which you want to plot histograms. Plot all the colums by default
    :param fit(str): you can plot the pdf of a the univariate distribution you want to fit. write a string corresponding to a univariate distribution ex "univariate-epispline"
    :return: creates a png file with the good histograms
    """
    df = pd.DataFrame.from_csv(path)
    if columns is None:
        columns = df.columns
    l = len(columns)

    if l<=2:
        display =121
    elif l<=4:
        display = 221
    elif l<=6:
        display = 231
    elif l<=8:
        display = 331



    for i in range(l):
        sample = list(df[columns[i]].dropna())
        plt.subplot(display+i)
        plt.hist(sample, n, normed=1, color='slateblue')
        plt.title(columns[i])
        if fit is not None:
            distr_class = distribution_factory(fit)
            distr = distr_class(sample)
            t = np.linspace(min(sample), max(sample),1000)
            l = len(t)
            f = np.ones(l)
            for i in range(l):
                f[i] = distr.pdf(t[i])
            plt.plot(t, f)
        plt.ylabel("Probability density")

    if fit is not None:
        title = path+'_'+fit+'_histogram.png'
    else:
        title = path+'_histogram.png'

    plt.savefig(title)
    plt.clf()


def create_files(n=n, copula_string='independence-copula', path=path,df=pd.DataFrame.from_csv(data_file),dimkeys=dimkeys, index=index, kind=kind, marginal_string=marginal_string, method=method, segment_marginal=segment_marginal):
    """
    This function creates csv files that correspond to dataframe we use.
    It creates U sample and P sample and return a column of S sample (cf documentation).

    :param n: size of the sample
    :param copula_string: The kind of copula you want to compute
    :param datatype : string whether it is 'actuals' or 'errors'
    :param index_diag : index of the diagonal where you want to project
    :return: Vector of the S value (cf latex for the meaning of U, O, P, Q,R, S
    """

    l = len(df.index)
    if method == 'wholeyear':
        dt_list = df.index
    elif method=='daytoday':
        dt_list = df.index[100:l-1]

    dimension = len(dimkeys)


    dict_epis_parameters = dict.fromkeys(dimkeys)
    if marginal_string == 'univariate-epispline':
        for i in dimkeys:
            if os.path.isfile(path + '/epispline_parameters_'+str(i)+'_' + marginal_data + '.csv'):
                dict_epis_parameters[i] = pd.DataFrame.from_csv(path + '/epispline_parameters_'+str(i)+'_' + marginal_data + '.csv')
            else:
                dict_epis_parameters[i] = pd.DataFrame(None, index=dt_list,
                                                   columns=['alpha', 'beta', 'N', 'w0', 'u0', 'delta'])

    diago = diag(dimension)

    suffix = return_suffix(kind, index, dimkeys, copula_string,segment_marginal)

    P_file_exists = os.path.isfile(path + '/samples/P_' + suffix)
    if P_file_exists:
        df_P = pd.DataFrame.from_csv(path + '/samples/P_' + suffix)
    else:
        df_U = dict.fromkeys(dimkeys)
        for i in dimkeys:
            U_file_exists = os.path.isfile(path + '/samples/U_' + copula_string)
            if U_file_exists:
                df_U[i] = pd.from_csv(path + '/samples/U_' + copula_string +'_'+str(i)+ '.csv')
            else:
                df_U[i] = pd.DataFrame(None, index=dt_list, columns=range(n))

        df_P = pd.DataFrame(None, index=dt_list, columns=range(n))
    S = []


    if method == 'wholeyear':

        input = dict.fromkeys(dimkeys)
        for i in dimkeys:
            input[i] = df[i].values.tolist()

        marginals = dict.fromkeys(dimkeys)
        for i in dimkeys:
            marginals[i] = marginal_from_input(marginal_string, input[i], None, method, dict_epis_parameters[i])

        distr_class = distribution_factory(copula_string)
        mydistr = distr_class(dimkeys, input)

    for mydt in dt_list:
        #print(mydt)
        if method == 'daytoday':

            input = dict.fromkeys(dimkeys)
            for i in dimkeys:
                input[i] = df[i].loc[df.index < mydt].values.tolist()

            if segment_marginal == 'segmented':

                input_segmented = dict.fromkeys(dimkeys)
                for i in dimkeys:
                    segmented_df = segmenter.OuterSegmenter(df.loc[df.index < mydt], df,
                                                            'copula_experiments/segment_input_wind_FH'+str(i)+'.txt',
                                                            mydt).retval_dataframe()
                    input_segmented[i] = segmented_df[i].values.tolist()
                input_marginals= input_segmented
            else:
                input_marginals=input


            marginals = dict.fromkeys(dimkeys)
            for i in dimkeys:
                marginals[i] = marginal_from_input(marginal_string, input_marginals[i], mydt, method, dict_epis_parameters[i])

            distr_class = distribution_factory(copula_string)
            mydistr = distr_class(dimkeys, input)

        if P_file_exists:
            P = df_P.loc[mydt].values
        else:
            if U_file_exists:
                U = np.transpose(df_U.loc[mydt].values)
            else:
                U = mydistr.generates_U(n)
                for j in range(dimension):
                    df_U[dimkeys[j]].loc[mydt] = U[:, j]

            P = diago.proj_scalar(U, index, kind)
            df_P.loc[mydt] = P

        O = df.loc[mydt].loc[dimkeys].values
        Q = np.zeros(dimension)
        for i in range(dimension):
            Q[i] = marginals[dimkeys[i]].cdf(O[i])

        R = diago.proj_scalar(Q, index, kind, mydistr)
        counter = sum(1 for i in range(n) if P[i] <= R)
        S.append(counter / n)


    if marginal_string == 'univariate-epispline':
        for i in dimkeys:
            dict_epis_parameters[i].to_csv(path + '/epispline_parameters_'+str(i)+'_' + marginal_data + '.csv')

    if not P_file_exists:
        df_P.to_csv(path + '/samples/P_' + suffix)
        for i in dimkeys:
            df_U[i].to_csv(path + '/samples/U_' + copula_string +'_'+str(i)+ '.csv')

    return S


def test_marginal(datatype=datatype,dps_index=0,distr_list=univariate_list,method=method,dps=dps,marginal_data=marginal_data,segment_marginal=segment_marginal):
    """
    This test will create dataframe on csv files and rank histograms to compare which univariate distribution fits well the marginals.
    :param datatype: 
    :param dps_index: 
    :param distr_list: 
    :param method: 
    :param dps: 
    :param marginal_data: 
    :return: 
    """


    df_data = pd.DataFrame.from_csv('copula_experiments/datas_BPA.csv')
    path = './copula_experiments/BPA/' + source + '_' + datatype + '_' + str(dps[0]) + '_' + str(dps[1]) + '_' + method

    print('Computing test_marginal : ' + source + ' ' + datatype + ' dps' + str(dps[dps_index])  + ' ' + method+' '+marginal_data+' '+segment_marginal)
    date_error = []
    if method=='wholeyear':
        l = len(df_data.index)
        if marginal_data == 'hour':
            dfatdps = df_data.loc[df_data.index.hour == dps[dps_index], datatype]
        elif marginal_data == 'anytime':
            dfatdps = df_data.loc[:, datatype]

        list_dates = dfatdps.dropna().index.date
        df = pd.DataFrame(None, index=list_dates, columns=distr_list)



        sample = list(dfatdps.dropna())

        l = len(dfatdps)

        for distr in distr_list:
            print(distr)
            distr_class = distribution_factory(distr)
            marginal = distr_class(sample)
            for mydate in list_dates:
                # Even thaugh we fitted the distribution with different hours (if marginal_data = anytime),
                # We must observe at dps.
                dfatdpsmydate = dfatdps.loc[dfatdps.index.date == mydate]
                df.loc[mydate, distr] = marginal.cdf(*dfatdpsmydate.loc[dfatdpsmydate.index.hour == dps[dps_index]].values)


    elif method=='daytoday':
        l = len(df_data.index)
        list_dates = pd.date_range(df_data.index[0]+dt.timedelta(days=100), end=df_data.index[l - 1]).date

        df = pd.DataFrame(None, index=list_dates, columns=distr_list)
        if marginal_data == 'hour':
            dfatdps = df_data[df_data.index.hour == dps[dps_index]]
        elif marginal_data == 'anytime':
            dfatdps = df_data


        for distr in distr_list:
            print(distr)
            if distr == 'univariate-epispline':
                if os.path.isfile(path + '/epispline_parameters_'+ str(dps[dps_index])+'_'+marginal_data+segment_marginal+'.csv'):
                    df_epis_parameters1 = pd.DataFrame.from_csv(path + '/epispline_parameters_' + str(dps[dps_index])+'_'+marginal_data+segment_marginal+'.csv')
                else:
                    df_epis_parameters1 = pd.DataFrame(None, index=list_dates,
                                                       columns=['alpha', 'beta', 'N', 'w0', 'u0', 'delta'])
            else:
                df_epis_parameters1 = None


            for mydate in list_dates:
                dfpast = dfatdps.loc[dfatdps.index.date < mydate]
                if segment_marginal=='segmented':
                    dfpast = segmenter.OuterSegmenter(dfpast, dfatdps,
                                                       'copula_experiments/segment_input_wind_' + datatype + '.txt',
                                                      dt.datetime(mydate.year, mydate.month, mydate.day,dps[dps_index])).retval_dataframe()

                input1 = list(dfpast.loc[:, datatype].dropna())

                marginal = marginal_from_input(distr,input1,mydate,method,df_epis_parameters1)

                # Even thaugh we fitted the distribution with different hours (if marginal_data = anytime),
                # We must observe at dps.
                dfatdpsmydate = dfatdps[dfatdps.index.date == mydate][datatype]

                temp = dfatdpsmydate[dfatdpsmydate.index.hour == dps[dps_index]].values
                try:
                    df.loc[mydate, distr] = marginal.cdf(*temp)
                except:
                    print('Data problem at ' + str(mydate))
                    date_error.append(mydate)

            if distr == 'univariate-epispline':
                df_epis_parameters1.to_csv(path + '/epispline_parameters_'+str(dps[dps_index])+'_'+marginal_data+segment_marginal+'.csv')

    df.to_csv(path + '/test_univariate_dps' + str(dps[dps_index]) + '_' + marginal_data +segment_marginal+ '.csv')
    csv_to_histogram(path + '/test_univariate_dps' + str(dps[dps_index]) + '_' + marginal_data +segment_marginal+ '.csv')

def marginal_from_input(distr=None,input=None,mydate=None,method='daytoday',dataframe=None):
    """
    This function takes an input and return a marginal object. It is very useful when you store tha parameters of EpiSplineDistribution and directly input them.
    It is not useful anymore except if one change the code in distributions.py to permit to input directly the parameter of an epispline without computing the optimization problem with pyomo and ipopt.
    :return:
    """

    if distr == 'univariate-epispline':
        if method == 'wholeyear':
            mydate = dataframe.index[len(dataframe.index) - 1]
        if any(math.isnan(y) for y in dataframe.loc[mydate]):
            # If a parameter is not set we have to fit the epispline distribution
            # And then we assign all the parameters
            marginal = UnivariateEpiSplineDistribution(input)
            dataframe['alpha'][mydate] = marginal.alpha
            dataframe['beta'][mydate] = marginal.beta
            dataframe['N'][mydate] = marginal.N
            dataframe['w0'][mydate] = marginal.w0
            dataframe['u0'][mydate] = marginal.u0
            dataframe['delta'][mydate] = marginal.delta
            for i in range(1, marginal.N + 1):
                dataframe.loc[mydate,'a' + str(i)] = marginal.a[i]
            return marginal
        else:
            # We impose our parameters that were already been calculated
            # So that the program will be faster
            return UnivariateEpiSplineDistribution(input, input_parameters=dataframe.loc[mydate])
    else:
        distr_class = distribution_factory(distr)
        return distr_class(input)


def return_suffix(kind=kind,index=index,dimkeys=dimkeys,copula_string=None,marginal_string = None,marginal_data=None,segment_marginal=None):
    """
    :return: a coherent string that permits to give a suffix in the name of our files.
    """

    if kind == 'diagonal':
        diago = diag(len(dimkeys))
        res = 'diag' + str(diago.list_of_diag[index])
    elif kind == 'marginal':
        res = 'dps' + str(dimkeys[index])
    elif kind=='kendall':
        res = 'kendall'

    if copula_string is not None:
        res = res+ '_' + copula_string

    if marginal_string is not None:
        res = res +'_'+marginal_string

    if marginal_data is not None:
        res = res +'_'+marginal_data

    if segment_marginal is not None:
        res = res +'_'+segment_marginal

    res = res + '.csv'

    return res


def compute_emd_results(bounds_list = bounds_list,datatype=datatype,source=source,dps=dps,method=method, marginal_string=marginal_string,segment_marginal=''):
    """
    Thanks to the S csv files. Calculate the EMD results (cf documentation diagonal_projection.pdf)
    The S files must be created before thanks to create_S on every diagonal.
    bounds_list : list of bounds where you want to compute the EMD for example if you want to focus on the tail the  bound would be (0, 0.1) and (0.9, 1) if you want to have a global vision it will be (0,1)
    """

    diag2 = diag(2)
    path = './copula_experiments/BPA/' + source + '_' + datatype + '_' + str(dimkeys)+'_'+method
    index = 0
    suffix = return_suffix(kind, index, dimkeys, marginal_string=marginal_string, marginal_data=marginal_data,segment_marginal=segment_marginal)
    df_S = pd.DataFrame.from_csv(path + '/S_' + suffix)
    copula_list = list(df_S.columns)
    list_of_diag = []

    for diagonal in diag2.list_of_diag:
        list_of_diag.append(str(diagonal))

    iterables = [list_of_diag,copula_list]
    index = pd.MultiIndex.from_product(iterables, names=['Diagonal', 'Copula'])
    df = pd.DataFrame(index=index, columns=bounds_list)

    for index in range(len(list_of_diag)):
        suffix = return_suffix(kind, index, dps, marginal_string=marginal_string, marginal_data=marginal_data)
        df_S = pd.DataFrame.from_csv(path+'/S_'+suffix)
        list_dates = df_S.index.tolist()
        l = len(list_dates)
        V = np.arange(l) / l

        for copula_string in copula_list:
            for low_bound, up_bound in bounds_list:
                df.loc[(list_of_diag[index],copula_string)][(low_bound, up_bound)] = emd_sort(df_S.loc[:, copula_string].values, V, low_bound,up_bound)
    if segment_marginal is not None:
        df.to_csv(path+'/emd_results_'+marginal_string+'_'+segment_marginal+'.csv')
    else:
        df.to_csv(path + '/emd_results_' + marginal_string + '.csv')

def print_contours(df=pd.DataFrame.from_csv(data_file),dimkeys=dimkeys,marginal_string = marginal_string,copula_list=copula_list,segment_marginal=None):
    """
    print the contours of the pdf of the copula object fitting data from df
    """
    df =df.convert_objects(convert_numeric=True)
    subset = []
    for i in dimkeys:
        subset.append(i)
        subset.append('FH' + i)
    df = df.dropna(axis=0, how='any', subset=subset)
    mydt = df.index[len(df.index)-1]

    input = dict.fromkeys(dimkeys)
    for i in dimkeys:
        input[i] = df[i].values.tolist()

    if segment_marginal == 'segmented':

        input = dict.fromkeys(dimkeys)
        for i in dimkeys:
            segmented_df = segmenter.OuterSegmenter(df.loc[df.index < mydt],df,
                                                    'copula_experiments/segment_input_wind_FH' + str(i) + '.txt',
                                                    mydt).retval_dataframe()
            input[i] = segmented_df[i].values.tolist()
    else:
        input = dict.fromkeys(dimkeys)
        for i in dimkeys:
            input[i] = df[i].loc[df.index < mydt].values.tolist()

    marginals = dict.fromkeys(dimkeys)
    for i in dimkeys:
        marg_class = distribution_factory(marginal_string)
        marginals[i] = marg_class(input[i])

    distr_class = distribution_factory(copula_string)
    mydistr = distr_class(dimkeys, input, marginals)


    xedges = np.arange(-100, 100)
    yedges = np.arange(-100, 100)

    H, xedges, yedges = np.histogram2d(input[dimkeys[0]],input[dimkeys[1]], bins=(xedges, yedges))
    H = H.T  # Let each row list bins with common y range.

    plt.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.xlabel('Error at '+str(dimkeys[0])+' forecast')
    plt.ylabel('Errors at '+str(dimkeys[1])+ ' forecast')
    plt.title('Histogram or errors')
    plt.savefig('./copula_experiments/BPA/histogram2d_errors.png')
    plt.clf()


    for mydistr in copula_list:
        print(mydistr)
        distr_class = distribution_factory(mydistr)
        copula = distr_class(dimkeys,input,marginals)


        x, y = np.meshgrid(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100))

        z = np.zeros((len(x),len(y)))
        tic()
        for i in range(len(x)):
            for j in range(len(y)):
                z[i][j]= copula.pdf({dimkeys[0]:x[i][j],dimkeys[1]:y[i][j]})
        toc()

        graphe = plt.contour(x, y, z, 10)
        plt.xlabel('Errors at '+str(dimkeys[0]))
        plt.ylabel('Errors at '+str(dimkeys[1]))
        plt.title('PDF Contours with '+mydistr+' and '+marginal_string+' marginals')
        plt.savefig('./copula_experiments/BPA/pdf_contours_'+mydistr+'_'+marginal_string+'.png')
        plt.clf()


if __name__ == '__main__':
    # Set the options.
    gosm_options.set_globals()
    #create_all_S()errors_dict['EH2']
    n = 1000
    datatype = 'EH1'
    kind = 'diagonal'
    source = 'wind'
    index= 0
    marginal_string = 'univariate-epispline'
    bounds_list = [(0, 1), (0, 0.1), (0.9, 1)]
    copula_list = ['gaussian-copula','student-copula','frank-copula', 'clayton-copula','independence-copula']
    univariate_list = ['univariate-uniform','univariate-normal','univariate-student',
                       'univariate-empirical','univariate-epispline']
    method = 'daytoday'
    marginal_data='hour'
    dps_index = 1
    segment_marginal = 'segmented'
    copula_string='clayton-copula'
    data_file = 'copula_experiments/datas_BPA_'+datatype+'_hours.csv'
    dimkeys =['12', '13']
    path = './copula_experiments/BPA/' + source + '_' + datatype + '_' + str(dimkeys) + '_' + method

    #creates_data()
    #creates_data_errors()
    #print_epispline_pdf_actuals()
    #print_errors_contours(marginal_string,copula_list)
    #csv_to_histogram('./copula_experiments/datas_BPA_all.csv',30,['actuals','EH1','normal-sample','uniform-sample'],'univariate-epispline')

    #compute_emd_results(bounds_list = bounds_list,datatype=datatype,source=source,dps=dps,method=method,marginal_string=marginal_string,segment_marginal=segment_marginal)
    print_contours(df=pd.DataFrame.from_csv(data_file), dimkeys=dimkeys, marginal_string=marginal_string,copula_list=copula_list,segment_marginal=segment_marginal)

    #df_to_hourdf('copula_experiments/datas_BPA_all.csv')
    #create_files(n=n,copula_string=copula_string,path=path,df=pd.DataFrame.from_csv(data_file),dimkeys=dimkeys,index=index,kind=kind,marginal_string=marginal_string,method=method,segment_marginal=segment_marginal)
    #create_S(n=n,datatype=datatype,kind=kind,source=source,dps=dps,index=index,copula_list=copula_list,marginal_string=marginal_string,method=method,marginal_data=marginal_data,segment_marginal=segment_marginal)
    #test_marginal(datatype=datatype, dps_index=dps_index, distr_list=univariate_list, method=method, dps=dps,marginal_data=marginal_data,segment_marginal=segment_marginal)
    #path = './copula_experiments/BPA/tests_H1_H2_' + source + '_' + datatype + '_' + str(dps[0]) + '_' + str(dps[1]) + '_' + method
    #suffix= return_suffix(kind,index,dps,marginal_string=marginal_string,marginal_data=marginal_data)

    #csv_to_histogram(path + '/S_test_errors_' + suffix)
