import numpy as np
import healpy as hp



band_off_region(pix,pix_center,size=np.radians(10),nside=64):
    
    theta_center, phi_center = hp.pix2ang(nside,pix_center)
    num_disc = np.pi*np.sin(theta_center)//size+1
    num_disc=int(num_disc)
    phi_disc_cent = np.linspace(phi_center,phi_center+2*np.pi,num_disc)
    theta_disc_cent = np.full(num_disc,theta_center)
    
    disc_vec = hp.ang2vec(theta_disc_cent,phi_disc_cent)
    disc_list = []
    for i in range(1,len(disc_vec)-1):
        discs = hp.query_disc(nside,disc_vec[i,:],size)
        disc_list.append(discs)
    disc_arr = np.concatenate(disc_list)
    in_off_region = np.isin(pix,disc_arr)
    return in_off_regio
