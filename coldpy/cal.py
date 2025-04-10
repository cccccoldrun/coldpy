from haversine import haversine
from haversine import Unit
import numpy as np
import numba 
import pandas as pd
from scipy.stats import t
# @numba.njit()
def cal_xx_yy(lats,lons,d):
    '''计算网格点之间的距离,输入经纬度
    返回点到（-90，0）的距离x距离和y距离'''
    
    
    x=[]
    y=[]
    
    for lat in lats:
        if (lat==90)|(lat==-90):
             x.append(0)
        else:
            a=haversine([lat,0], [lat,d], unit='m')
            x.append(a) 
        
        a=haversine([lat,0], [-90,0], unit='m')
        y.append(a)
    
    xx=np.array(x).reshape(len(lats),1).repeat(len(lons),axis=1).cumsum(axis=1)
    yy=np.array(y).reshape(len(lats),1).repeat(len(lons),axis=1)
    
    return xx,yy

@numba.njit()
def dp(surface_pressures,levels)  : 
    '''参考ncl的delta_xxx方法计算每层之间的气压厚度
    输入surface_pressures和levels,返回每层的气压厚度(time,lev,lat,lon)'''
    pressure_lev=np.sort(levels)
    print(pressure_lev)
    pressure_top = np.min(pressure_lev)
    lt,llat,llon=surface_pressures.shape
    
    out=np.full_like(surface_pressures,np.nan,dtype=np.float32).reshape(lt,1,llat,llon)
    delta_pressures=np.repeat(out,len(levels)).reshape(lt,-1,llat,llon)
    print(delta_pressures.shape)
    for t in np.arange(lt):
        for lat in np.arange(llat):
            for lon in np.arange(llon):
                surface_pressure=surface_pressures[t,lat,lon]
                print(surface_pressure)
                if np.isnan(surface_pressure):continue
                delta_pressure = np.full_like(pressure_lev, np.nan,dtype=np.float32)
                indices=np.where(pressure_lev <= surface_pressure)[0]
                
                start_level = min(indices) ##top
                end_level = max(indices) ##bottom
                
                if np.less(surface_pressure,max(pressure_lev)) :
                    mid=(pressure_lev[end_level]+pressure_lev[end_level+1])/2
                    if np.less(mid,surface_pressure):
                        end_level = max(indices) + 1##bottom

                for i in np.arange(start_level,end_level+1):
                    if i==start_level:
                        delta_pressure[start_level] = (pressure_lev[start_level] + pressure_lev[start_level + 1]) / 2 - pressure_top
                    elif i==end_level:
                        delta_pressure[end_level] = surface_pressure - (pressure_lev[end_level] + pressure_lev[end_level - 1]) / 2
                    else:
                        delta_pressure[i] = (pressure_lev[i+1]-pressure_lev[i-1])/2
                delta_pressures[t,:,lat,lon]=delta_pressure
    return delta_pressures

import numpy as np
import numba 
from scipy.stats import t
@numba.njit()
def corr_3d(x,y):
    ##计算相关系数3dx3d
    '''3维数组和3维数组的相关系数
    返回r,t,p'''
    shape=x.shape
    out1=np.zeros(shape=(shape[1],shape[2]))
    out2=np.zeros(shape=(shape[1],shape[2]))
    p=np.zeros(shape=(shape[1],shape[2]))
    for lat in range(shape[1]):
        for lon in range(shape[2]):
            a=x[:,lat,lon]
            b=y[:,lat,lon]
            
            mean_a = np.sum(a) / len(a)
            mean_b = np.sum(b) / len(b)

            # 计算偏离均值的差的乘积和每个变量的差的平方
            sum_ab = np.sum((a - mean_a) * (b - mean_b))
            sum_a2 = np.sum((a - mean_a) ** 2)
            sum_b2 = np.sum((b - mean_b) ** 2)
            try:
                r = sum_ab / np.sqrt(sum_a2 * sum_b2)
            except:
                print(sum_ab,sum_a2,sum_b2)
                r=np.nan
            n = len(a)

            # 计算t统计量
            t_statistic_manual = r * np.sqrt((n - 2) / (1 - r**2))
            
            out1[lat,lon]=r
            out2[lat,lon]=t_statistic_manual
    # p=t_test(out2,df=shape[0]-2)
    return out1,out2


@numba.njit()
def corr(x,y):
    ##计算相关系数1dx3d,返回r,t,p
    a=x
    shape=y.shape
    out1=np.zeros(shape=(shape[1],shape[2]))
    out2=np.zeros(shape=(shape[1],shape[2]))
    for lat in range(shape[1]):
        for lon in range(shape[2]):
            b=y[:,lat,lon]
            
            mean_a = np.sum(a) / len(a)
            mean_b = np.sum(b) / len(b)

            # 计算偏离均值的差的乘积和每个变量的差的平方
            sum_ab = np.sum((a - mean_a) * (b - mean_b))
            sum_a2 = np.sum((a - mean_a)**2)
            sum_b2 = np.sum((b - mean_b)**2)
            try:
                r = sum_ab / np.sqrt(sum_a2 * sum_b2)
            except:
                r=np.nan
            n = len(a)

            # 计算t统计量
            t_statistic_manual = r * np.sqrt((n - 2) / (1 - r**2))
            
            out1[lat,lon]=r
            out2[lat,lon]=t_statistic_manual
            
    # p=t_test(out2,df=shape[0]-2)
    
    return out1,out2

@numba.njit
def regression(x, y):
    ##计算回归1dx3d,返回回归系数和t统计量，p和预测值
    n, m, p = y.shape
    slopes = np.zeros((m, p))
    t_stats = np.zeros((m, p))
    pred=np.zeros((n, m, p))
    
    X_mean = np.mean(x)
    
    for i in range(m):
        for j in range(p):
            Y_slice = y[:, i, j]
            Y_mean = np.mean(Y_slice)
            
            # 计算回归系数 (slope) 和截距 (intercept)
            try:
                slope = np.sum((x - X_mean) * (Y_slice - Y_mean)) / np.sum((x - X_mean) ** 2)
                intercept = Y_mean - slope * X_mean
                
                # 计算预测值
                y_pred = slope * x + intercept
                
                pred[:,i,j]=y_pred
                # 计算残差
                residual = Y_slice - y_pred
                # residuals[:, i, j] = residual
                
                # 残差平方和
                RSS = np.sum(residual ** 2)
                
                # 总平方和
                TSS = np.sum((Y_slice - Y_mean) ** 2)
                
                # R-squared
                R_squared = 1 - RSS / TSS
                
                # 样本数量
                s_xx = np.sum((x - X_mean) ** 2)
                std_err_slope = np.sqrt(RSS / (n - 2) / s_xx)
                
                # t 统计量
                t_stat = slope / std_err_slope
                
                # 存储结果
                slopes[i, j] = slope
                # intercepts[i, j] = intercept
                # R_squareds[i, j] = R_squared
                t_stats[i, j] = t_stat
            except:
                slopes[i, j] = np.nan
                # intercepts[i, j] = intercept
                # R_squareds[i, j] = R_squared
                t_stats[i, j] = np.nan
                
    # p=t_test(t_stats,df=n-2)        
    return slopes, t_stats, pred

@numba.njit
def regression_4d(x, y):
##计算回归1dx4d
    n, l,m, p = y.shape
    slopes = np.zeros((l,m, p))
    t_stats = np.zeros((l,m, p))
    pred=np.zeros((n,l, m, p))
    
    X_mean = np.mean(x)
    for ll in range(l):
        for i in range(m):
            for j in range(p):
                Y_slice = y[:,ll, i, j]
                Y_mean = np.mean(Y_slice)
                
                # 计算回归系数 (slope) 和截距 (intercept)
                try:
                    slope = np.sum((x - X_mean) * (Y_slice - Y_mean)) / np.sum((x - X_mean) ** 2)
                    intercept = Y_mean - slope * X_mean
                    
                    # 计算预测值
                    y_pred = slope * x + intercept
                    
                    pred[:,ll,i,j]=y_pred
                    # 计算残差
                    residual = Y_slice - y_pred
                    # residuals[:, i, j] = residual
                    
                    # 残差平方和
                    RSS = np.sum(residual ** 2)
                    
                    # 总平方和
                    TSS = np.sum((Y_slice - Y_mean) ** 2)
                    
                    # R-squared
                    R_squared = 1 - RSS / TSS
                    
                    # 样本数量
                    s_xx = np.sum((x - X_mean) ** 2)
                    std_err_slope = np.sqrt(RSS / (n - 2) / s_xx)
                    
                    # t 统计量
                    t_stat = slope / std_err_slope
                    
                    # 存储结果
                    slopes[ll,i, j] = slope
                    # intercepts[i, j] = intercept
                    # R_squareds[i, j] = R_squared
                    t_stats[ll,i, j] = t_stat
                except:
                    slopes[ll,i, j] = np.nan
                    # intercepts[i, j] = intercept
                    # R_squareds[i, j] = R_squared
                    t_stats[ll,i, j] = np.nan
                
    # p=t_test(t_stats,df=n-2)        
    return slopes, t_stats, pred

def regression_3d_3d(x, y):
    """
    计算x与y每个二维切片的回归系数及t统计量，支持三维数组。
    
    参数：
    x: 自变量三维数组，形状为 (n, m, p)
    y: 因变量三维数组，形状为 (n, m, p)
    
    返回：
    slopes: 回归系数二维数组，形状为 (m, p)
    intercepts: 截距二维数组，形状为 (m, p)
    R_squareds: 拟合优度二维数组，形状为 (m, p)
    t_stats: t统计量二维数组，形状为 (m, p)
    residuals: 残差三维数组，形状为 (n, m, p)
    """
    n, m, p = y.shape
    slopes = np.full((m, p), np.nan)
    t_stats = np.full((m, p), np.nan)
    pred=np.zeros((n, m, p))
    for i in range(m):
        for j in range(p):
            # 使用切片提取当前二维切片
            x_slice = x[:, i, j]  # 取出 x 在 (i,j) 的所有 n 值
            y_slice = y[:, i, j]  # 取出 y 在 (i,j) 的所有 n 值
            
            x_mean=x_slice.mean()
            y_mean=y_slice.mean()
            try:
            # 计算均值
                slope = np.sum((x_slice - x_mean) * (y_slice - y_mean)) / np.sum((x_slice - x_mean) ** 2)
                intercept = y_mean - slope * x_mean
                
                # 计算预测值
                y_pred = slope * x_slice + intercept
                
                pred[:,i,j]=y_pred
                # 计算残差
                residual = y_slice - y_pred
                # residuals[:, i, j] = residual
                
                # 残差平方和
                RSS = np.sum(residual ** 2)
                
                # 总平方和
                TSS = np.sum((y_slice - y_mean) ** 2)
                
                # R-squared
                R_squared = 1 - RSS / TSS
                
                # 样本数量
                s_xx = np.sum((x_slice - x_mean) ** 2)
                std_err_slope = np.sqrt(RSS / (n - 2) / s_xx)
                
                # t 统计量
                t_stat = slope / std_err_slope
                
                # 存储结果
                slopes[i, j] = slope
                t_stats[i, j] = t_stat
            except:
                slopes[i, j] = np.nan
                t_stats[i, j] = np.nan
    # p = t_test(t_stats, df=n-2)        
    return slopes, t_stats  ,pred

def t_test(t_stats,df):
    ##t-test
    ps=(1-t.cdf(abs(t_stats),df))*2
    return ps

def running_average(data, window_size):
    """
    滑动平均
    1维数组
    Calculate the running average of a 1D array.
    
    Parameters:
        data (array-like): Input data series.
        window_size (int): Size of the moving window.
        
    Returns:
        numpy.ndarray: Running average.
    """
    return pd.Series(data).rolling(window=window_size, center=True).mean().to_numpy()

def running_correlation(x, y, window_size):
    """
    滑动相关
    Calculate the running correlation coefficient between two 1D arrays.
    
    Parameters:
        x (array-like): First data series.
        y (array-like): Second data series (must be the same length as x).
        window_size (int): Size of the moving window.
        
    Returns:
        numpy.ndarray: Running correlation coefficients.
    """
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length.")
    
    # Convert to Pandas Series for rolling correlation
    x_series = pd.Series(x)
    y_series = pd.Series(y)
    
    return x_series.rolling(window=window_size, center=True).corr(y_series).to_numpy()