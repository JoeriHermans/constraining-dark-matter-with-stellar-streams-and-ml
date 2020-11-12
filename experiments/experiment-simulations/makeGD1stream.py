import os, os.path
import pickle
import numpy
from optparse import OptionParser
import gd1_util
from galpy.util import save_pickles
from galpy.util import bovy_conversion
from galpy.potential import MWPotential2014
import astropy.units as u


#code to generate streams
def get_options():
    usage = "usage: %prog [options]"
    parser = OptionParser(usage=usage)

    #set timpact chunks
    parser.add_option("-t","--timpacts",dest='timpacts',default="64sampling",
                      help="Impact times in Gyr to consider; should be a comma separated list")

    parser.add_option("--chunk_size",dest='chunk_size',default=64,
                      type='int',
                      help="size of subsets of full timpacts")

    parser.add_option("--td",dest='td',default=5.,
                      type='float',
                      help="tdisrupt in Gyr")

    parser.add_option("--prog",dest='prog',default=-40.,
                      type='float',
                      help="current progenitor location in phi1, based on this the GD1 model will be implemented, works with -40 only at the moment")

    #parser.add_option("--sigv",dest='sigv',default=0.32,
    #                  type='float',
    #                  help="vel disp in km/s")

    parser.add_option("--leading_arm",action="store_false",dest='leading_arm',default=True,
                      help="leading or trailing arm")

    parser.add_option("--plummer",action="store_false",
                      dest="plummer",default=True,
                      help="If set, use a Plummer DM profile rather than Hernquist")

    parser.add_option("--tind",dest='tind',default=0,
                      type='int',
                      help="index of timpact chunk")


    return parser

ro=8.
vo=220.

parser= get_options()
options,args= parser.parse_args()

########setup timpact chunks

print ("td=%.2f"%options.td)

def parse_times(times,age,ro=ro,vo=vo):
    if 'sampling' in times:
        nsam= int(times.split('sampling')[0])
        return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
                for ti in numpy.arange(1,nsam+1)/(nsam+1.)*age]
    return [float(ti)/bovy_conversion.time_in_Gyr(vo,ro)
            for ti in times.split(',')]

timpacts= parse_times(options.timpacts,options.td,ro=ro,vo=ro)
size= options.chunk_size
tarray=[timpacts[i:i+size] for i  in range(0, len(timpacts), size)]
timpactn=tarray[options.tind]

print (timpactn)

prog=options.prog

if prog == 0. :
    new_orb_lb = None
    isob=0.8

elif prog == -40. :
    new_orb_lb=[188.04928416766532, 51.848594007807456, 7.559027173643999, 12.260258757214746, -5.140630283489461, 7.162732847549563]
    isob=0.45

elif prog == -20. :
    new_orb_lb=[154.1678010489945, 60.148998301606596, 8.062181685320652, 12.005853651554805, 0.9876589841345033, -126.24003922281933]
    isob=0.45


if  options.leading_arm:
    lead_name='leading'
    lead=True

else :
    lead_name='trailing'
    lead=False

if options.plummer:
    hernquist=False
    sub_type = 'Plummer'
else :
    hernquist=True
    sub_type = 'Hernquist'

sigv_age=numpy.round((0.3*3.2)/options.td,2)
print ("velocity dispersion is set to %f"%sigv_age)

sdf_smooth=gd1_util.setup_gd1model(age=options.td,new_orb_lb=new_orb_lb,isob=isob,leading=lead,sigv=sigv_age)

pepperfilename= 'gd1_pepper_{}_{}_sigv{}_td{}_{}sampling_progphi1{}_MW2014_{}.pkl'.format(lead_name,sub_type,sigv_age,options.td,len(timpacts),prog,options.tind)

sdf_pepper=gd1_util.setup_gd1model(timpact=timpactn,age=options.td,hernquist=hernquist,new_orb_lb=new_orb_lb,isob=isob,leading=lead,length_factor=1.,sigv=sigv_age)

save_pickles(pepperfilename,sdf_smooth,sdf_pepper)
