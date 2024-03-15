

import astropy
from astropy.modeling import models, fitting
from astropy.coordinates import Angle
import astropy.io.fits as FF
import numpy as np
import os
import pylab as pl
import scipy.interpolate as II

import astropy.units as uu
from astropy.units import Quantity

def read_atik_file(fname):
    """ Read atik file from fname. Assume a Bayer filter pattern and correct for it. 
        Returns [3, Nx, Ny] pixel array.
    """
    
    ff = FF.open(fname)

    dd = ff[0].data

    b = dd[:-1:2, :-1:2]
    g = dd[1::2,  :-1:2]
    r = dd[:-1:2,  1::2]

    return np.array((r,g,b)).astype(np.float64)


# From Stack Overflow
def radial_profile(data, center, plot_check=False):
    """ Returns the radial profile of the image in a 1d array """
    x,y = np.indices((data.shape)) # first determine radii of all pixels
    r = np.sqrt((x-center[0])**2+(y-center[1])**2)
    ind = np.argsort(r.flat) # get sorted indices
    sr = r.flat[ind] # sorted radii
    sim = data.flat[ind] # image values sorted by radii
    ri = sr.astype(np.int32) # integer part of radii (bin size = 1)
    # determining distance between changes
    deltar = ri[1:] - ri[:-1] # assume all radii represented
    rind = np.where(deltar)[0] # location of changed radius
    nr = rind[1:] - rind[:-1] # number in radius bin
    csim = np.cumsum(sim, dtype=np.float64) # cumulative sum to figure out sums for each radii bin
    tbin = csim[rind[1:]] - csim[rind[:-1]] # sum for image values in radius bins
    radialprofile = tbin/nr # the answer


    if plot_check:
        rs = np.arange(len(radialprofile))
        pl.figure(figsize=(15,7))
        pl.subplot(211)
        pl.plot(data[center[0], :], lw=2, color='black')
        pl.plot(rs + center[1], radialprofile, color='red')
        pl.plot(-rs + center[1], radialprofile, color='red')
        pl.title("Horizontal Crosscut")
        pl.ylabel("Counts")
        pl.xlim(0, data.shape[1])
        pl.grid(True)

        pl.subplot(212)
        pl.plot(data[:, center[1]], lw=2, color='black')
        pl.plot(rs + center[0], radialprofile, color='red')
        pl.plot(-rs + center[0], radialprofile, color='red')
        pl.title("Vertical Crosscut")
        pl.xlabel("Pixel Number")
        dx = data.shape[0]-data.shape[1]
        pl.xlim(dx/2, data.shape[1]+dx/2)
        pl.ylabel("Counts")
        pl.grid(True)

        pl.tight_layout()

    return radialprofile

def find_ring_center_helper(img, res, x, y):
    if res[x,y] == 0.0: 
        res[x,y] = max(radial_profile(img, (x,y)))


def find_ring_center(img, NIterMax=200, verbose=False, plot_check=False):
    """ Find the ring center by doing a gradient search around the middle 
    
    Args:
        img[Nx,Ny]: The image to search for the ring center. The search
            box starts at Nx/2, Ny/2

        NIterMax: Number of iterations allowed

    Returns:
        {"peak_x": Peak X pixel,
        "peak_y": Peak Y pixel, 
        "res": The search results (higher value = more contrasty ring)

    """

    res = np.zeros_like(img)

    x,y = img.shape[0]//2, img.shape[1]//2
    iter = 0

    if plot_check: pl.figure()
    res[x,y]     = max(radial_profile(img, (x,y)))
    if verbose:
        print("%10s %5s %5s %6s" % ("Iter", "X", "Y", "Value"))

    while iter < NIterMax:
        if verbose: print("%10i %5i %5i %6.0f" % (iter, x, y, res[x,y]))
        find_ring_center_helper(img, res, x+1,y+1)
        find_ring_center_helper(img, res, x+1,y-1)
        find_ring_center_helper(img, res, x-1,y+1)
        find_ring_center_helper(img, res, x-1,y-1)
        find_ring_center_helper(img, res, x+1,y)
        find_ring_center_helper(img, res, x-1,y)
        find_ring_center_helper(img, res, x,y+1)
        find_ring_center_helper(img, res, x,y-1)



        if plot_check: 
            pl.clf()
            pl.imshow(res)

        peak_x, peak_y = np.unravel_index(np.argmax(res), res.shape)

        if (peak_x == x) and (peak_y == y):
            return {"peak_x": peak_x, "peak_y": peak_y, "res": res}
        x,y = peak_x, peak_y
        iter += 1

    


def compute_ring_values(img, center):
    """ Computes the ring diameter and FWHM [pix]

    Args:
        img: The image to compute on
        center: The center pixel (x,y)

    """

    rp = radial_profile(img, center)

    peak_loc = np.argmax(rp)

    if rp[peak_loc] < 500:
        print("WARNING: The Peak Value of the Ring Doesn't seem Right")

    gauss_initial = models.Gaussian1D(amplitude=rp[peak_loc],
                                        mean=peak_loc,
                                        stddev=20,
                                        bounds={"amplitude": [0, 1e6],
                                            "mean": [0,700],
                                            "stddev": [0, 100]})
    fitter = fitting.LevMarLSQFitter()
    x = np.arange(len(rp))
    res = fitter(gauss_initial, x, rp)

    return {"fwhm": res.fwhm, "radius": res.mean.value, "res": res}

def res_to_pair(x):
    return [x["peak_x"], x["peak_y"]]

def pix2mm(pix, imagescale=6.45e-3*2*uu.mm):
    return pix*imagescale

@uu.quantity_input
def coneangle_to_fno(theta: Quantity[uu.rad]):
    """ Convert full cone angle in radians to fno """

    return 1/(2*np.tan(theta/2))

def find_center_and_get_params(img):

    res = find_ring_center(img)
    cc = res_to_pair(res)
    fit = compute_ring_values(img, cc)

    return fit, cc
    

def compute_distance(i1, i2, spacer_width=4.00*uu.mm, verbose=True,
        imagescale=6.45e-3*2):
    """ Computes the distance from the fiber to the detector """

    fit1,_ = find_center_and_get_params(i1)
    fit2,_ = find_center_and_get_params(i2)

    diams = [fit1["radius"] * 2, fit2["radius"] * 2]

    DD = np.abs(diams[0]-diams[1])

    Dmax = np.max(diams)
    Dmin = np.min(diams)
    
    fiber_to_large_ring_distance = spacer_width * Dmax/DD

    fiber_to_small_ring_distance = fiber_to_large_ring_distance \
                                    - spacer_width

    RingRadius = pix2mm(Dmax/2)
    cone_diam = np.rad2deg(2*np.arctan(RingRadius/fiber_to_large_ring_distance))


    fwhm1_mm = pix2mm(fit1["fwhm"])
    fwhm2_mm = pix2mm(fit1["fwhm"])
    fwhm1_deg = (2*np.arctan(fwhm1_mm/2/fiber_to_small_ring_distance)).to(uu.deg)
    fwhm2_deg = (2*np.arctan(fwhm2_mm/2/fiber_to_small_ring_distance)).to(uu.deg)

    ss = f"""
    fiber->small: {fiber_to_small_ring_distance.value:3.2f} | {Dmin:3.1f} pix | Cone Diam: {cone_diam.value:2.3f} {cone_diam.unit}
    fiber->large: {fiber_to_large_ring_distance.value:3.2f}

    FWHM (small): {fwhm1_deg.value:3.5f} {fwhm1_deg.unit}
    FWHM (large): {fwhm2_deg.value:3.5f} {fwhm2_deg.unit}

    """

    with open("distances.txt", "w") as f:
        f.write(ss)
                
    if verbose:
        print(ss)

    return fiber_to_small_ring_distance, fiber_to_large_ring_distance, Dmin, cone_diam, fwhm1_deg, fwhm2_deg



def RED(img):
    return img[0]

def process_directory(path, plot_check=False):

    dirs = os.listdir(path)

    print(path)

    bg = RED(read_atik_file(os.path.join(path, "bg.fit")))
    z0 = RED(read_atik_file(os.path.join(path, "z0.fit")))-bg
    z4 = RED(read_atik_file(os.path.join(path, "z4.fit")))-bg

    dists = compute_distance(z0, z4)
    
    dist = dists[0]

    res_rad = []
    res_fwhm = []
    if plot_check:
        pl.figure(figsize=(16,16))
        pl.clf()

    for i in range(1,6):
        aa = RED(read_atik_file(os.path.join(path, f"a{i}.fit")))-bg
        res, cc = find_center_and_get_params(aa)
        fwhm = pix2mm(res["fwhm"])
        radius = pix2mm(res["radius"])

        radius_ang = (2*np.arctan(radius/dist)).to(uu.deg)
        fwhm_ang = (2*np.arctan(fwhm/2/dist)).to(uu.deg)
        print(coneangle_to_fno(radius_ang), fwhm_ang, radius_ang, fwhm_ang/radius_ang)

        res_rad.append(radius_ang)
        res_fwhm.append(fwhm_ang)

        if plot_check:
            pl.subplot(3,2,i)
            pl.imshow(aa)
            xcenter, ycenter = cc
            circ = pl.Circle((ycenter, xcenter), res["radius"], lw=0.1, fill=False, color='black')
            pl.axhline(xcenter, color='black', lw=1)
            pl.axvline(ycenter, color='black', lw=1)
            pl.title(f"FWHM: {fwhm_ang.value:1.3} Rad: {radius_ang.value:3.4} Distance: {dist.value:4.5}")
            pl.gca().add_patch(circ)

    pl.subplot(3,2,6)
    pl.plot([t.value for t in res_rad], [t.value for t in res_fwhm], lw=3, color='black')
    pl.axvline(14.6)
    pl.axhline(2.4)
    pl.xlabel("Cone angle [deg]")
    pl.ylabel("FWHM [deg] of ring")
    pl.ylim(0,5)
    pl.xlim(5,25)
    pl.tight_layout()

    return {"fib_det_dist": dist, "ring_radii": res_rad, "ring_fwhm": res_fwhm}

def process_tree(inpath, outpath):
    """ Go through all the subdirs in inpath """


    dirs = os.listdir(inpath)

    print(dirs)
    pl.ioff()

    for dir in dirs:
        if dir[0] != "S":
            print(f"Skipping {dir}")
            continue

        outname = os.path.join(outpath, f"{dir}.npy")
        print(dir, outname)

        if os.path.exists(outname):
            print(f"{outname} exists, skipping")
            continue

        res = process_directory(os.path.join(inpath, dir), plot_check=True)

        np.save(outname, [res], allow_pickle=True)
        pl.savefig(os.path.join(outpath, f"{dir}.pdf"))



def report(inpath):

    dirs = os.listdir(inpath)

    allres = []
    dists = []
    fwhm_3_9 = []
    
    for dir in dirs:
        if not dir.endswith("npy"): continue

        res = np.load(os.path.join(inpath, dir), allow_pickle=True)[0]

        xsd = res["ring_radii"]
        ysd = res["ring_fwhm"]

        xs = [t.value for t in xsd]
        ys = [t.value for t in ysd]

        dists.append(res["fib_det_dist"])

        allres.append([xs,ys])

        ff = II.interp1d(xs, ys, bounds_error=False, fill_value=0)
        f = ff(15.0)

        fwhm_3_9.append(f)

    dds = np.array([t.value for t in dists])
    ok = (dds>11) & (dds < 14)

    pl.figure(3)
    pl.clf()
    fwhm_3_9 = np.array(fwhm_3_9)

    # Following fitted from Claire's paper. see note
    f = lambda x: -0.04497*x + 1.006
    finv = lambda xinv: 1.006/0.04497 - 1/0.04497*xinv
    pl.hist(fwhm_3_9[ok], 50, align='mid', color='black')
    pl.axvline(3.0, color='black')
    secax = pl.gca().secondary_xaxis('top', functions=(f,finv))
    secax.set_xlabel("EE_3.57")
    pl.xlabel("FWHM @ f/3.9 [deg]")
    pl.text(0, 10, "0= code errors")
    N = np.count_nonzero(fwhm_3_9[ok]>3)
    pl.text(4,10,"N_fail = %s" % N)
    N = np.count_nonzero(fwhm_3_9[ok])
    pl.ylabel("Count")
    pl.title("Histogram of FWHM @ f/3.9")
    pl.text(2,20,"Total= %s" % N)

    pl.figure(2)
    pl.clf()
    
    pl.hist(dds, 50, color='black')
    pl.axvline(11)
    pl.axvline(14)
    pl.xlabel("Distance from Fiber -> Detector [mm]")
    pl.ylabel("Number")
    pl.title("Stddev in OK region: %1.2f [mm]" % (np.std(dds[ok])))


    pl.figure(1)
    pl.clf()
    pl.axvline(14.6, lw=3, color='black')
    allres = np.array(allres)
    for x,y in allres[ok]:
        pl.plot(x,y, lw=2, color='black', alpha=.15)

    pl.xlabel("Cone Angle [deg]")
    pl.ylabel("Ring FWHM [deg]")

    return dds, allres
        




