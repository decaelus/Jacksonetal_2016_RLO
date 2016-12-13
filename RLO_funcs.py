import scipy.optimize as so
import numpy as np

from astropy.units import cds
#Now to calculate mass loss rates
from astropy import constants as const
import astropy.units as u

from scipy.integrate import dblquad

#molecular mass of hydrogen gas
mu = 2.*1.00794*u.g/u.mol
#atomic mass of hydrogen molecule
amu = 2.*u.u
#atomic mass of hydrogen atom
#amu = 1.*u.u
gas_const = const.R

def calc_xcm(q, a=1.):
    return a * 1./(q + 1.)

def calc_omegasq(q, a=1.):
    return (q + 1.)/a**3.

def calc_smallq_roche_lobe(q, a=1.):
    return (q/3.)**(1./3)*a

def calc_smallq_roche_limit(q, Rp=0.1):
    return (q/3.)**(-1./3)*Rp

def calc_approx_rlobe(mass_ratio, a=1.): 
    q = mass_ratio**(1./3)
    
    #Here's what actually appears in Eggleton 1983, apj 268:368-369
    return a*0.49*q*q/(0.6*q*q + np.log(1. + q))

#potential
def potroche(r, q, a=1., theta=0., phi=0.):
    
    omegasq = calc_omegasq(q, a=a)
    xcm = calc_xcm(q, a=a)
  
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    sphi = np.sin(phi)
    cphi = np.cos(phi)
    
    #$x = r \cos \theta$
    #$y = r \sin \theta \cos \phi$
    #$z = r \sin \theta \sin \phi$
    
    d2 = np.sqrt( a**2 - 2.*r*a*cth + r**2 )

    #Roche potential
    pot = -q/r - 1./d2 - 0.5 * omegasq * ( (r*sth*cphi)**2 + (r*cth - xcm)**2 )

    #derivative of Roche potential
    pot_r = -q/r**2 + (a*cth - r)/d2**3 + omegasq * ( (r*sth*cphi) + cth*(r*cth - xcm) )

    return pot, pot_r

#d(potential)/dr
def dpotroche_dr(r, q, a=1., theta=0., phi=0.):
    return potroche(r, q, a=a, theta=theta, phi=phi)[1]

def find_Lagrange_surface(q, x0 = None):
    if(x0 is None):
#        x0 = calc_smallq_roche_lobe(q)
        x0 = calc_approx_rlobe(q)

    return so.newton(dpotroche_dr, x0, args=(tuple([q])))
#    return so.bisect(dpotroche_dr, 0., 1., args=(q))

def calc_xL1(mass_ratio, x0=None, a=1.): 
    if(isinstance(mass_ratio, float)):    
        return find_Lagrange_surface(mass_ratio, x0=x0)
    else:
        ret_val = np.zeros_like(mass_ratio)
        for i in range(len(ret_val)):
            ret_val[i] = find_Lagrange_surface(mass_ratio[i], x0=x0)
        return ret_val
    
#Arras's volume-averaged potential 
def Arras_phi(rv, q, a=1.):
#    Arras_phi(rL1, q)*const.G*ma/a
    rfac = rv/a
    return (1./rfac)*(q + 1./3*rfac**3.*(1. + q) + 4./45*((1. + q)**2./q + 3.*(1. + q)/q + 9./q)*rfac**6.)
    
#Returns period or semi-major axis, given the other
#  P in days, a in AU, and Mstar in solar masses
def Keplers_Third_Law(P=None, a=None, Mstar=1.*cds.Msun):

    bigG = 4.*np.pi*np.pi/cds.Msun*(cds.au*cds.au*cds.au)/(cds.yr*cds.yr)

    if((P is None) and (a is None)):
        raise ValueError("P or a must be given!")
    elif(a is not None):
        return (a*a*a/(bigG*Mstar/(4.*np.pi*np.pi)))**(1./2)
    elif(P is not None):
        return (P*P*(bigG*Mstar/(4.*np.pi*np.pi)))**(1./3)

#Now to calculate Ritter's gamma function and compare it to ours and a more precise numerical solution

from scipy.misc import derivative
import os.path

def ritters_gamma(q):
    inv_q = 1./q
    
    if(isinstance(q, float)):
        if((0.04 <= inv_q) & (inv_q < 1.)):
            return 0.954 + 0.025*np.log(inv_q) - 0.038*(np.log(inv_q))**2
        elif((1. <= inv_q) & (inv_q < 20.)):
            return 0.954 + 0.039*np.log(inv_q) + 0.114*(np.log(inv_q))**2
    else:
        ret_val = np.zeros_like(inv_q)
        ind = (0.04 <= inv_q) & (inv_q < 1.)
        ret_val[ind] = 0.954 + 0.025*np.log(inv_q[ind]) - 0.038*(np.log(inv_q[ind]))**2

        ind = (1. <= inv_q) & (inv_q < 20.)
        ret_val[ind] = 0.954 + 0.039*np.log(inv_q[ind]) + 0.114*(np.log(inv_q[ind]))**2
    
        return ret_val

def calc_approx_gamma(q):
    if(isinstance(q, float)):
        xL1 = calc_xL1(q)
        potL1 = potroche(xL1, q)[0]
        volL1 = calc_pot_vol(q, potL1)
        rvL1 = calc_rv(volL1)
        
        eff_grav = q/rvL1**2
        
        return derivative(approx_pot_rv, rvL1, args=tuple([q]), dx=1e-6, order=5)/eff_grav
    else:
        ret_val = np.array([])
        for cur_q in q:
            xL1 = calc_xL1(cur_q)
            potL1 = potroche(xL1, cur_q)[0]
            volL1 = calc_pot_vol(cur_q, potL1)
            rvL1 = calc_rv(volL1)
        
            eff_grav = cur_q/rvL1**2
        
            ret_val = np.append(ret_val, derivative(approx_pot_rv, rvL1, args=tuple([cur_q]), dx=1e-6, order=5)/eff_grav)
            
        return ret_val

def pot_vol_minus_given_pot(pot, q, vol):
    return calc_pot_vol(q, pot) - vol

def calc_pot_from_vol(vol, q):

    r0 = 1e-6
    pot0 = potroche(r0, q)[0]

    r1 = calc_xL1(q)
    pot1 = potroche(r1, q)[0]

    return so.brentq(pot_vol_minus_given_pot, pot0, pot1, args=tuple([q, vol]))

def ritters_bigF(q):
    inv_q = 1./q
    
    if(isinstance(q, float)):
        if (0.5 <= inv_q) & (inv_q < 10.):
            return 1.23 + 0.5*np.log10(inv_q)
    else:    
        ret_val = np.zeros_like(inv_q)

        ind = (0.5 <= inv_q) & (inv_q < 10.)
        ret_val[ind] = 1.23 + 0.5*np.log10(inv_q[ind])
    
        return ret_val

def nozzle_area(q, vthermal, orb_freq):
    A = A_approx(q)    
    return 2.*np.pi*(vthermal/orb_freq)**2./np.sqrt(A*(A - 1.))

def calc_rvL1(q):
    xL1 = calc_xL1(q)
    potL1 = potroche(xL1, q)[0]
    
    volL1 = calc_pot_vol(q, potL1)
    return calc_rv(volL1)

def calc_num_gamma(q):
    rvL1 = calc_rvL1(q)
    vol_arr = np.linspace(volL1*(1. - 1e-3), volL1, 11)
    pot_arr = np.array([calc_pot_from_vol(vol_arr[i], q) for i in range(len(vol_arr))])

    rv_arr = calc_rv(vol_arr)
 
    eff_grav = q/rvL1**2

    drv = np.gradient(rv_arr)
    dpot_drv_L1 = np.gradient(pot_arr, drv, edge_order=2)[-1]

    return dpot_drv_L1/eff_grav

def calc_num_F(q):
    rvL1 = calc_rvL1(q)
    
    A = A_num(q)
    
    omegasq = calc_omegasq(q)
    
    return (1./omegasq)/np.sqrt(A*(A - 1.))*(q/rvL1**3)

def eggletons_rvL1(q, a=1):
    return a*0.49*q**(2./3)/(0.6*q**(2./3) + np.log(1. + q**(1./3)))

def calc_approx_F(q):
    A = A_approx(q)
    
    omegasq = calc_omegasq(q)
    
    rvL1 = eggletons_rvL1(q)
    
    return (1./omegasq)/np.sqrt(A*(A - 1.))*(q/rvL1**3)

#Comparing approximate expression for rv, Eqn \ref{eq:potential_vs_volume_radius} to more exact solution
def approx_pot_rv(rv, q, a=1.):
    return -(1./a + 1./(2.*(q + 1))) - q/rv*(1. + (1./3)*(q + 1.)/q*(rv/a)**3 + \
                                            4./45*(((q + 1.) + 3.*(q + 1.) + 9)/q**2)*(rv/a)**6)

def potroche_minus_given_pot(r, q, pot, theta, phi, a=1.):
    return (potroche(r, q, a=a, theta=theta, phi=phi)[0] - pot)

def approx_pot_rv_minus_given_pot(r, q, pot):
    return approx_pot_rv(r, q) - pot

#CAUTION! This routine won't work if the desired r > L1 point!
def find_r_pot(q, pot, theta, phi, a=1.):
    r0 = abs(q/pot)
    r1 = calc_xL1(q, a=a)
    
    return so.brentq(potroche_minus_given_pot, r0, r1, args=tuple([q, pot, theta, phi]))
    
def r3(phi, theta, q, pot):
    return (1./3)*find_r_pot(q, pot, theta, phi)**3*np.sin(theta)
    
def gfun(theta):
    return 0.

def hfun(theta):
    return 2.*np.pi

#Returns the volume of a given potential surface
def calc_pot_vol(q, pot):
    
    return dblquad(r3, 0., np.pi, gfun, hfun, args=(q, pot))[0]
    
def calc_rv(vol):
    return (3./(4.*np.pi)*vol)**(1./3)

def mass_center(q, a=1.):
    return a*(1./(1. + q))

def A_num(q, x0=None, a=1.):

    xL1 = calc_xL1(q, x0=x0, a=a)
    xcm = mass_center(q)
    
    return a**3/(q + 1.)*(q/(xL1)**3 + 1./(abs(xL1 - a)**3))

def A_approx(q):
    b1 = 2.*3.**(2./3)
    b2 = b1/4. - 2.
    
    return 4. + b1/(b2 + q**(1./3) + 1/q**(1./3))

def return_amu(T):
    #2016 Nov 4 - Checks whether temperature exceeds thermal dissociation
    #  temperature for hydrogen, ~2,000 K

    diss_temperature = 2000.

    if(T.to('K').value >= diss_temperature):
        return 1.*u.u
    else:
        return 2.*u.u

def calc_scale_height(Mp, Rp, Tp):
    eff_grav = const.G*Mp/Rp**2
    
    return const.k_B*Tp/(return_amu(Tp)*eff_grav)

def calc_scale_height_with_tides(Ms, a, Mp, Rp, Tp):
    q = (Mp.to('kg')/Ms.to('kg')).value
    rph_over_a = (Rp.to('m')/a.to('m')).value
    eff_grav = np.abs(potroche(rph_over_a, q)[1]*const.G*Ms/a**2)
       
    return const.k_B*Tp/(return_amu(Tp)*eff_grav)

def calc_photosphere_density(Mp, Rp, Tp, tau_eq=0.56, kappa=1e-2*u.cm**2/u.g):
    #Using relation between optical depth from 
    #  Howe & Burrows (2012 -- http://iopscience.iop.org/0004-637X/756/2/176/article)
    #  and taking opacity from 
    #  Li+ (2012 -- http://www.nature.com/nature/journal/v463/n7284/extref/nature08715-s1.pdf)
    
    H = calc_scale_height(Mp, Rp, Tp)
    return tau_eq/(kappa*np.sqrt(2.*np.pi*Rp*H))

def calc_photosphere_density_with_tides(Ms, a, Mp, Rp, Tp, tau_eq=0.56, kappa=1e-2*u.cm**2/u.g):
    #Using relation between optical depth from 
    #  Howe & Burrows (2012 -- http://iopscience.iop.org/0004-637X/756/2/176/article)
    #  and taking opacity from 
    #  Li+ (2012 -- http://www.nature.com/nature/journal/v463/n7284/extref/nature08715-s1.pdf)
    
    H = np.abs(calc_scale_height_with_tides(Ms, a, Mp, Rp, Tp))
    
    return tau_eq/(kappa*np.sqrt(2.*np.pi*Rp*H))

def calc_photosphere_density_hot(Mp, Rp, Tp):
#2015 Aug 24 -- From Phil's suggestion, we're calculating this density a different way now.
#
#2015 Aug 31 -- Although this approach might be interesting and helpful, it is not consistent
#  with what appears in MESA, so I'm reverting to the original approach, above.

    #From Tramell+ (2011), the UV cross-section at 13.6 eV for hydrogen ionization
    sigma = 6.3e-18*u.cm**2 #cm^2
    column_density = 1./sigma #column density for optical depth unity
    
    H = calc_scale_height(Mp, Rp, Tp)

    return (column_density*return_amu(Tp)/H)

def Weiss_density(Rp):
    #Returns planetary density as given by Weiss & Marcy (2014) ApJL 783, L6.
    #
    #Rp -- planetary radius (Earths)

    #unitless
    Rgeo = 6371.*u.km
    
    my_Rp = Rp.to('km').value/(Rgeo.to('km').value)
    
    first_dividing_line = 1.5
    second_dividing_line = 4.

    Earths_density = 5.3*u.g/u.cm**3

    if(my_Rp < first_dividing_line):
        density = (2.43 + 3.39*my_Rp)*u.g/u.cm**3
    elif((my_Rp >= first_dividing_line) and (my_Rp < second_dividing_line)):
        density = (2.69*my_Rp**(-2.07))*Earths_density
    else:
        density = None
    
    return density
        
#average effective radiative temperature
def calc_Teff(a, Rs, Ts):
    return np.sqrt(Rs/a/np.sqrt(2.))*Ts

def calc_vthermal(T, amu=None):
    #molecular hydrogen's adiabatic index
#    gamma = 1.4
    #molecular hydrogen's adiabatic index
#    gamma = 5./3
#    return np.sqrt(gamma*const.k_B*T/amu)

    #Isothermal sound speed -- This is what Phil gave me, but it looks wrong!
    #It's actually the RMS velocity -- 
    #  https://en.wikipedia.org/wiki/Root-mean-square_speed.
#   return np.sqrt(3.*const.k_B*T/amu)

    if(amu is None):
        return np.sqrt(const.k_B*T/return_amu(T))
    else:
        return np.sqrt(const.k_B*T/amu)

def calc_exp(md, ma, a, rph, densityph, vthermal, delta_pot=None):
    #mass ratio
    q = (md.to('kg')/ma.to('kg')).value
    rph_over_a = (rph.to('km')/a.to('km')).value

    #2016 Oct 26 - Must convert volume-average photospheric potential to the 
    #  volume-equiv radius, i.e. do NOT use the transit radius of the planet!
    #  The MESA code uses the volume-equivalent radius for the binary calcs.
    rv_ph = Arras_rv(rph_over_a, q)
    rL1 = eggletons_rvL1(q)

#   Make sure to get the units right
    potL1 = Arras_phi(rL1, q)*const.G*ma/a
    potph = Arras_phi(rv_ph, q)*const.G*ma/a

    if(delta_pot == None):
        delta_pot = potL1 - potph          
    
    return np.exp((delta_pot.to('erg/g')/((vthermal.to('cm/s'))**2)).value)
    
##  2016 Sep 23 -- USING THE WRONG POTENTIAL!!

##  2016 Aug 30 -- Retiring these old equations
#   xL1 = calc_xL1(q)
#   potph = potroche(rph_over_a, q, theta=np.pi/2.)[0]*const.G*ma/a
#   potL1 = potroche(xL1, q)[0]*const.G*ma/a
#   if(delta_pot == None):
#       delta_pot = potL1 - potph        

##  Sign flips compared to Arras's potential because of how the potentials are defined
#   return np.exp(-(delta_pot.to('erg/g')/((vthermal.to('cm/s'))**2)))

def Mdot_Arras(md, ma, a, rph, densityph, vthermal, delta_pot=None):
    #mass ratio
    q = (md.to('kg')/ma.to('kg')).value

    #2016 Nov 1 - Use the volume-equivalent for Rp, NOT Rp itself
    rph_over_a = rph.to('km').value/a.to('km').value
    rv_ph = Arras_rv(rph_over_a, q)*a
       
    #square of the orbital frequency
    orb_freq = np.sqrt(calc_omegasq(q)*const.G*ma/a**3)
    exp = calc_exp(md, ma, a, rv_ph, densityph, vthermal, delta_pot=delta_pot)
    return -np.exp(-1./2)*densityph * exp * (vthermal.to('cm/s')) * nozzle_area(q, vthermal, orb_freq)
    
def calc_Hp0(Td, rvL1, md):
    return gas_const*Td*rvL1**2./(mu*const.G*md)
    
def Mdot_Ritter(md, ma, a, rph, densityph, Td, mu=mu, delta_r=None):  
    q = (md.to('kg')/ma.to('kg')).value
    
    #volume-radius for Roche lobe
    rvL1 = eggletons_rvL1(q, a=a)

    F = ritters_bigF(q)
    
    M0_dot = 2.*np.pi*np.exp(-1./2)*(gas_const*Td/mu)**(3./2)*rvL1**3./(const.G*md)*densityph*F
    
    Hp0 = calc_Hp0(Td, rvL1, md)
    Hp = Hp0/ritters_gamma(q)

    if(delta_r == None):
        delta_r = (rvL1.to('km') - rph.to('km'))
    
    return -M0_dot*np.exp(-delta_r.to('km')/Hp.to('km'))

def calc_Rstar(mass):
    #very simple mass-radius relation
    return (mass.to('solMass').value)**0.62*u.solRad

def calc_Tstar(mass):
    #very simple mass-luminosity relation
    return (mass.to('solMass').value)**0.65*(6000.*u.K)

#Based directly on the Fortran version of Phil's model
def Arras_fortran(md, ma, a, rph, densityph, Td):
    grav = const.G*md/rph**2
    hp = const.k_B*Td/(return_amu(Td)*grav)
    v_th = calc_vthermal(Td)
    
    q = md.to('g').value/ma.to('g').value
    sep = a
    Omega = np.sqrt(calc_omegasq(q)*const.G*ma/sep**3.)
    rvL1 = eggletons_rvL1(q)*a
    
    phiL1 = const.G*md/rvL1 \
          + 1./3*(const.G*(md + ma)*rvL1**2/sep**3.) \
          + 4./45*const.G*md*rvL1**5/sep**6*((md + ma)**2 + 3.*ma*(md + ma) + 9.*ma**2)/md**2

    phi = const.G*md/rph \
          + 1./3*(const.G*(md + ma)*rph**2/sep**3.) \
          + 4./45*const.G*md*rph**5/sep**6*((md + ma)**2 + 3.*ma*(md + ma) + 9.*ma**2)/md**2

    my_ritter_exponent = ((phiL1.to('m^2/s^2').value - phi.to('m^2/s^2').value)/(v_th.to('m/s').value)**2)
    rhoL1 = densityph/np.sqrt(np.exp(1.)) * np.exp( my_ritter_exponent )
    
    q13=q**(1./3.)
    Asl = 4. + 4.16/(-0.96 + q13 + 1./q13)
    area = 2. * np.pi * (v_th.to('m/s')/Omega)**2 / np.sqrt( Asl*(Asl-1.0) )
    
    my_mdot_thin = -rhoL1 * v_th * area
    return my_mdot_thin

def dadt_tidal(Mp, Ms, Rs, a, Qs):
    return -9./2*np.sqrt(const.G/Ms)*Mp*Rs**5/Qs*a**(-11./2)

def dMpdt_tidal(Mp, Ms, Rs, a, Qs):
    return -1./2*dadt_tidal(Mp, Ms, Rs, a, Qs)/a*Mp

def Ktide(Mp, Rp, Ms, a):
    q = (Mp.to('kg').value)/(Ms.to('kg').value)
    xi = (calc_smallq_roche_lobe(q, a=a).to('km').value)/(Rp.to('km').value)
    
    return 1. - 3./(2.*xi) + 1./(2.*xi**2.)

#From Ribas+ (2006)
def Fx(age, a):
    lamb = 29.7*u.erg/u.s/u.cm**2
    bet = -1.23
    
    return lamb*(age.to('year').value/1e9)**bet/(a.to('AU').value)**2

def dMpdt_evap(Mp, Ms, Rs, a, Rp, Fxuv, epsilon=0.1):
    Kt = Ktide(Mp, Rp, Ms, a)
       
    return np.pi*Rp**3.*epsilon*Fxuv/(const.G*Mp*Kt)

def Arras_rv3(r0, q, a=1.):
    return r0**3*(1. + (r0/a)**3*(1. + 1./q) + \
                  4./5*(r0/a)**6*(2.*(1. + 1./q)**2 + \
                                  1./q*(1. + 1./q) + 3./q)) 

def Arras_rv(r0, q, a=1.):
    return (Arras_rv3(r0, q, a=a))**(1./3)
