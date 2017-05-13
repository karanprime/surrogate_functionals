# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:29:41 2017

@author: ray
"""




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec

def meshgrid3(x,y,z):
    """ Create a three-dimensional meshgrid """
 
    nx = len(x)
    ny = len(y)
    nz = len(z)

    xx = np.swapaxes(np.reshape(np.tile(x,(1,ny,nz)),(nz,ny,nx)),0,2)
    yy = np.swapaxes(np.reshape(np.tile(y,(nx,1,nz)),(nx,nz,ny)),1,2)
    zz = np.tile(z,(nx,ny,1))
    
    return xx,yy,zz

class  DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps. 
       Created by Joe Kington and submitted to StackOverflow on Dec 1 2012
       http://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete
    """
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 1)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):

        xy = self.poly.xy
        xy[2] = val, 1
        xy[3] = val, 0
        self.poly.xy = xy
  
        # Suppress slider label
        self.valtext.set_text('')

        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.iteritems():
            func(val)





class slice3_noScatter(object):
    def __init__(self,xx,yy,zz,u):
        '''
        u: PCA analysis color (R,G,B)
        
        '''
        
        self.x = xx[:,0,0]
        self.y = yy[0,:,0]
        self.z = zz[0,0,:]
        
        self.data = u
        

        self.fig = plt.figure(1,(20,7))

        self.ax1 = self.fig.add_subplot(2,3,1,aspect='equal')
        self.ax2 = self.fig.add_subplot(2,3,2,aspect='equal')
        self.ax3 = self.fig.add_subplot(2,3,3,aspect='equal')

        self.xplot_zline = self.ax1.axvline(color='m',linestyle='--',lw=2)
        self.xplot_zline.set_xdata(self.z[0]) 

        self.xplot_yline = self.ax1.axhline(color='m',linestyle='--',lw=2)
        self.xplot_yline.set_ydata(self.y[0])

        self.yplot_xline = self.ax2.axhline(color='m',linestyle='--',lw=2)
        self.yplot_xline.set_ydata(self.x[0])

        self.yplot_zline = self.ax2.axvline(color='m',linestyle='--',lw=2)
        self.yplot_zline.set_xdata(self.z[0])

        self.zplot_xline = self.ax3.axvline(color='m',linestyle='--',lw=2)
        self.zplot_xline.set_xdata(self.x[0])

        self.zplot_yline = self.ax3.axhline(color='m',linestyle='--',lw=2)
        self.zplot_yline.set_ydata(self.y[0])

#        self.PCAscatter = self.ax0.scatter(PCA_x, PCA_y, color = PCA_u)#, alpha=0.5)
        self.xslice = self.ax1.imshow(u[0,:,:,:],extent=(self.z[0],self.z[-1],self.y[0],self.y[-1]))
        self.yslice = self.ax2.imshow(u[:,0,:,:],extent=(self.z[0],self.z[-1],self.x[0],self.x[-1]))
        self.zslice = self.ax3.imshow(u[:,:,0,:],extent=(self.x[0],self.x[-1],self.y[0],self.y[-1]))

        # Create and initialize x-slider
        self.sliderax1 = self.fig.add_axes([0.125,0.08,0.225,0.03])
        self.sliderx = DiscreteSlider(self.sliderax1,'',0,len(self.x)-1,increment=1,valinit=0)
        self.sliderx.on_changed(self.update_x)
        self.sliderx.set_val(0)

        # Create and initialize y-slider
        self.sliderax2 = self.fig.add_axes([0.4,0.08,0.225,0.03])
        self.slidery = DiscreteSlider(self.sliderax2,'',0,len(self.y)-1,increment=1,valinit=0)
        self.slidery.on_changed(self.update_y)
        self.slidery.set_val(0)

        # Create and initialize z-slider
        self.sliderax3 = self.fig.add_axes([0.675,0.08,0.225,0.03])
        self.sliderz = DiscreteSlider(self.sliderax3,'',0,len(self.z)-1,increment=1,valinit=0)
        self.sliderz.on_changed(self.update_z)
        self.sliderz.set_val(0)

        z0,z1 = self.ax1.get_xlim()
        x0,x1 = self.ax2.get_ylim()
        y0,y1 = self.ax1.get_ylim()
        self.ax1.set_aspect((z1-z0)/(y1-y0))
        self.ax2.set_aspect((z1-z0)/(x1-x0))
        self.ax3.set_aspect((x1-x0)/(y1-y0))




    def xlabel(self,*args,**kwargs):
        self.ax2.set_ylabel(*args,**kwargs)
        self.ax3.set_xlabel(*args,**kwargs)

    def ylabel(self,*args,**kwargs):
        self.ax1.set_ylabel(*args,**kwargs)
        self.ax3.set_ylabel(*args,**kwargs)
  
    def zlabel(self,*args,**kwargs):
        self.ax1.set_xlabel(*args,**kwargs)
        self.ax2.set_xlabel(*args,**kwargs) 

    def update_x(self,value): 
        self.xslice.set_data(self.data[value,:,:])  
        self.yplot_xline.set_ydata(self.x[value])
        self.zplot_xline.set_xdata(self.x[value])

    def update_y(self,value): 
        self.yslice.set_data(self.data[:,value,:])  
        self.xplot_yline.set_ydata(self.y[value])
        self.zplot_yline.set_ydata(self.y[value])

    def update_z(self,value): 
        self.zslice.set_data(self.data[:,:,value])  
        self.xplot_zline.set_xdata(self.z[value])
        self.yplot_zline.set_xdata(self.z[value])


    def show(self):
        plt.show()



#if __name__ == '__main__':
#
#    # Number of x-grid points
#    nx = 100
#
#    # Number of 
#    ny = 100
#    nz = 200
#
#    x = np.linspace(-4,4,nx)
#    y = np.linspace(-4,4,ny)
#    z = np.linspace(0,8,nz)
#
#    xx,yy,zz = meshgrid3(x,y,z)
#    
##    result = 
#    # Display three cross sections of a Gaussian Beam/Paraxial wave
#    u = np.real(np.exp(-(2*xx**2+yy**2)/(.2+2j*zz))/np.sqrt(.2+2j*zz))
#    v = np.real(np.exp(-(3*xx**2+yy**2)/(.3+2j*zz))/np.sqrt(.5+2j*zz))
#    w = np.real(np.exp(-(4*xx**2+yy**2)/(.3+2j*zz))/np.sqrt(.6+2j*zz))
#    result = np.ones_like(u).tolist()
#    for index,x in np.ndenumerate(u):
#        result[index[0]][index[1]][index[2]] = (u[index[0]][index[1]][index[2]], v[index[0]][index[1]][index[2]], w[index[0]][index[1]][index[2]])
##    print u.shape
##    print u
#    print np.asarray(result).shape
#    s3 = slice3(xx,yy,zz,np.asarray(result))
#    s3.xlabel('x',fontsize=18)
#    s3.ylabel('y',fontsize=18)
#    s3.zlabel('z',fontsize=18)
#     
#
#    s3.show()
 
