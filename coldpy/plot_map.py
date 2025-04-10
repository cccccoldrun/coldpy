import cartopy.feature as cfeature
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.path import Path 
from cartopy.mpl.patch import geos_to_path
plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]

def plot_map(ax,
             shapes=[],
             shapes_width=[1],
             extent=[20,140,-20,60],
             deltax=30,
             deltay=20,
             COASTLINE=True,
             COASTLINE_width=0.8,
             labelsize=9,
             scale="50m"):
    """ax 为要画的图的ax,
    shapes是要添加的shapes,必须为cfeature.ShapelyFeature修饰后的比如:
    shape=Reader('shapefile_path').geometries()
    shape=cfeature.ShapelyFeature(shape, ccrs.PlateCarree(), edgecolor='k', facecolor='none')
    extent是要画的范围,
    dx和dy是经纬度的间隔,
    COASTLINE是是否添加海岸线,
    labelsize是标签大小
    scale是海岸线的分辨率,"""
    if len(shapes)!=0:
        if len(shapes_width)!=len(shapes):
            for shape in shapes:
                ax.add_feature(shape, linewidth=shapes_width[0])
        else:
            for i in range(len(shapes)):
                ax.add_feature(shapes[i], linewidth=shapes_width[i])

    if COASTLINE==True:
        ax.add_feature(cfeature.COASTLINE.with_scale(scale),lw=COASTLINE_width)

    ax.set_extent(extent,crs=ccrs.PlateCarree())
    ax.set_xticks(range(extent[0],extent[1]+1,deltax))#指定要显示的经纬度                      
    ax.set_yticks(range(extent[2],extent[3]+1,deltay))    
    ax.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式                       
    ax.yaxis.set_major_formatter(LatitudeFormatter()) 
    ax.tick_params(axis='both',which='major',labelsize=labelsize,direction='out',length=5,width=1,pad=2,top=True,right=True)
    return ax

def clip(ax,cn,geo,ctype="contourf"):
    """
    ax为axes对象,使用geopandas读取
    cn为要裁剪的对象,
    geo为裁剪区域,ctype为要裁剪的对象类型
    """
    clip_path=Path.make_compound_path(*geos_to_path(geo.geometry.to_list()))
    if ctype=="contourf" or ctype=="contour":
        for i in cn.collections:
            i.set_clip_path(clip_path,transform=ax.transData)
    elif ctype=="pcolormesh" or ctype=="quiver":
        cn.set_clip_path(clip_path,transform=ax.transData)
    else:
        cn.set_clip_path(clip_path,transform=ax.transData)