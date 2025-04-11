import numpy as np
from TimeCal.TC_Data import TC_Data

class TC_Calibrator():
    """A water cherenkov detector timing calibrator"""

    def __init__(self, wcd):
        self.wcd = wcd
        self.raw_data = None
        self.ref_mpmt = 0
        self.ref_mpmt_delay = 0.
        self.ref_mpmt_delay_sigma = 0.1
        self.epsilon_mean = 0.
        self.epsilon_sigma = 100.
        self.alpha_mean = 0.
        self.alpha_sigma = 100.
        self.epsilon_apply = False
        self.alpha_apply = True

    def assign_data(self, raw_data):
        """Assign the raw transit time data to the calibrator"""
        self.raw_data = raw_data

    def set_reference_mpmt(self, ref_mpmt, ref_mpmt_delay, ref_mpmt_delay_sigma):
        """Set the reference mPMT for the calibration
        ref_mpmt: integer between 0 and n_mpmt-1
        ref_mpmt_delay: (ns) the reference mPMT delay
        ref_mpmt_delay_sigma: (ns) sets the strength of this prior - a smaller value gives a stronger prior"""
        self.ref_mpmt = ref_mpmt
        self.ref_mpmt_delay = ref_mpmt_delay
        self.ref_mpmt_delay_sigma = ref_mpmt_delay_sigma

    def set_priors(self, epsilon_mean, epsilon_sigma, alpha_mean, alpha_sigma, epsilon_apply: bool = False, alpha_apply: bool = True):
        """Set the prior estimates for the calibration to break degeneracies of the linear equations
        units are ns, and the sigma specifies the strength of the prior. The _apply flags specify
        whether to apply the prior"""
        self.epsilon_mean = epsilon_mean
        self.epsilon_sigma = epsilon_sigma
        self.alpha_mean = alpha_mean
        self.alpha_sigma = alpha_sigma
        self.epsilon_apply = epsilon_apply
        self.alpha_apply = alpha_apply

    def calibrate(self, return_chisqs=False, place_info='design'):
        """With raw transit time data, perform a calibration, and save calibrations as
        estimates for the device constants: prop_est"""

        wcd = self.wcd
        raw_data = self.raw_data
        ref_mpmt = self.ref_mpmt
        ref_mpmt_delay = self.ref_mpmt_delay
        ref_mpmt_delay_sigma = self.ref_mpmt_delay_sigma
        e_mean = self.epsilon_mean
        e_sigma = self.epsilon_sigma
        e_apply = self.epsilon_apply
        a_mean = self.alpha_mean
        a_sigma = self.alpha_sigma
        a_apply = self.alpha_apply

        vc = wcd.light_velocity / wcd.prop_design['refraction_index']
        n_mpmt = len(wcd.mpmts)
        m_pmt = len(wcd.mpmts[ref_mpmt].pmts)
        w_led = min(6,len(wcd.mpmts[ref_mpmt].leds)) # At most 6 flashing LED locations
        chisqs = {}
        devs = {}
        dists = {}
        
        bad_list = []

        # The set of linear equations are indexed by r (LEDs), q (mPMTs), and p (PMTs)
        # If there is no data for a particular LED, then the equation is not included
        # in the set of linear equations. The following dictionary is used to indicate
        # if data exists.

        data_exists = {'led':{}, 'mpmt':{}, 'pmt':{}}

        # The coefficients are combinations of weights and delta-times

        ws = [[[[0. for k in range(m_pmt)] for j in range(n_mpmt)] for l in range(w_led)] for i in range(n_mpmt)]
        wdts = [[[[0. for k in range(m_pmt)] for j in range(n_mpmt)] for l in range(w_led)] for i in range(n_mpmt)]

        # Loop over all possible combinations of LED, mPMT, and PMT
        for i in range(n_mpmt):
            for l in range(w_led):
                r = i*w_led + l
                if l < len(wcd.mpmts[i].leds) and wcd.mpmts[i].leds[l] is not None: # some mPMTs have no LEDs
                    led = wcd.mpmts[i].leds[l]
                    # check if placement information is available (estimates are only available if it exists)
                    device_place = getattr(led, 'place_' + place_info, None)
                    if device_place is not None and 'loc' in device_place:
                        place = led.get_placement(place_info)
                        led_loc = place['location']
                        for j in range(n_mpmt):
                            for k in range(m_pmt):
                                p = j*m_pmt + k

                                t, t_sig = raw_data.get(i, l, j, k)
                                if t is not None:
                                    data_exists['led'][r] = True
                                    data_exists['mpmt'][j] = True
                                    data_exists['pmt'][p] = True

                                    wght = np.power(t_sig, -2)
                                    ws[i][l][j][k] = wght

                                    pmt = wcd.mpmts[j].pmts[k]
                                    # check if placement information is available (estimates are only available if it exists)
                                    device_place = getattr(pmt, 'place_' + place_info, None)
                                    if device_place is not None and 'loc' in device_place:
                                        place = pmt.get_placement(place_info)
                                        pmt_loc = place['location']
                                        light_vec = np.subtract(pmt_loc, led_loc)
                                        dist = np.linalg.norm(light_vec)
                                        dt = (t - dist / vc)

                                        wdts[i][l][j][k] = wght*dt

        # form linear equations to be solved: calculate coefficients for
        # each parameter - parameters are e, d, a
        aa, bb = [], []

        # The LED equations (epsilons):

        for i in range(n_mpmt):
            for l in range(w_led):
                r = i * w_led + l   # r is the index of the LED equation
                e_sum = [0.] * n_mpmt * w_led
                d_sum = [0.] * n_mpmt
                a_sum = [0.] * n_mpmt * m_pmt
                bb_sum = 0.
                for j in range(n_mpmt):
                    for k in range(m_pmt):
                        p = j * m_pmt + k
                        wght = ws[i][l][j][k]
                        if wght > 0.:
                            e_sum[r] += wght
                            d_sum[i] += wght
                            d_sum[j] -= wght
                            a_sum[p] += wght
                            bb_sum += wdts[i][l][j][k]

                if e_apply:
                    we = np.power(e_sigma, -2)
                    e_sum[r] -= we
                    bb_sum -= e_mean * we

                aa_row = e_sum + d_sum + a_sum
                aa.append(aa_row)
                bb.append(bb_sum)

        # The mPMT equations (deltas):

        for q in range(n_mpmt):
            e_sum = [0.] * n_mpmt * w_led
            d_sum = [0.] * n_mpmt
            a_sum = [0.] * n_mpmt * m_pmt
            bb_sum = 0.
            if q == ref_mpmt:
                wdel = np.power(ref_mpmt_delay_sigma, -2)
                d_sum[q] -= wdel
                bb_sum -= ref_mpmt_delay * wdel

            for l in range(w_led):
                r = q * w_led + l
                for j in range(n_mpmt):
                    if j != q:
                        for k in range(m_pmt):
                            p = j * m_pmt + k
                            wght = ws[q][l][j][k]
                            if wght > 0.:
                                e_sum[r] += wght
                                d_sum[q] += wght
                                d_sum[j] -= wght
                                a_sum[p] += wght
                                bb_sum += wdts[q][l][j][k]
                for i in range(n_mpmt):
                    if i != q:
                        r = i * w_led + l
                        for k in range(m_pmt):
                            p = q * m_pmt + k
                            wght = ws[i][l][q][k]
                            if wght > 0.:
                                e_sum[r] -= wght
                                d_sum[i] -= wght
                                d_sum[q] += wght
                                a_sum[p] -= wght
                                bb_sum -= wdts[i][l][q][k]

            aa_row = e_sum + d_sum + a_sum
            aa.append(aa_row)
            bb.append(bb_sum)

        # The PMT equations (alphas):

        for j in range(n_mpmt):
            for k in range(m_pmt):
                p = j * m_pmt + k
                e_sum = [0.] * n_mpmt * w_led
                d_sum = [0.] * n_mpmt
                a_sum = [0.] * n_mpmt * m_pmt
                bb_sum = 0.
                for i in range(n_mpmt):
                    for l in range(w_led):
                        r = i * w_led + l
                        wght = ws[i][l][j][k]
                        if wght > 0.:
                            e_sum[r] += wght
                            d_sum[i] += wght
                            d_sum[j] -= wght
                            a_sum[p] += wght
                            bb_sum += wdts[i][l][j][k]

                if a_apply:
                    wa = np.power(a_sigma, -2)
                    a_sum[p] -= wa
                    bb_sum -= a_mean * wa

                aa_row = e_sum + d_sum + a_sum
                aa.append(aa_row)
                bb.append(bb_sum)

        # Remove equations and parameters for which there is no data
        # Do this backwards, to simplify indexing

        for p in range(n_mpmt*m_pmt-1, -1, -1):
            if not data_exists['pmt'].get(p, False):
                index = n_mpmt * w_led + n_mpmt + p
                for row in aa:
                    row.pop(index)
                aa.pop(index)
                bb.pop(index)

        for q in range(n_mpmt-1, -1, -1):
            if not data_exists['mpmt'].get(q, False):
                index = n_mpmt * w_led + q
                for row in aa:
                    row.pop(index)
                aa.pop(index)
                bb.pop(index)

        for r in range(n_mpmt*w_led-1, -1, -1):
            if not data_exists['led'].get(r, False):
                for row in aa:
                    row.pop(r)
                aa.pop(r)
                bb.pop(r)

        # Solve the linear equations

        a = np.array(aa)
        b = np.array(bb)
        pars = np.linalg.solve(a, b)

        # Store pars in the parameter estimates
        # Only store the parameters for which there is data

        rr = 0
        for i in range(n_mpmt):
            for l in range(w_led):
                r = i * w_led + l
                if data_exists['led'].get(r, False):
                    wcd.mpmts[i].leds[l].prop_est['delay'] = pars[rr]
                    rr += 1

        qq = 0
        for j in range(n_mpmt):
            if data_exists['mpmt'].get(j, False):
                wcd.mpmts[j].prop_est['clock_offset'] = pars[rr + qq]
                qq += 1

        pp = 0
        for j in range(n_mpmt):
            for k in range(m_pmt):
                p = j * m_pmt + k
                if data_exists['pmt'].get(p, False):
                    wcd.mpmts[j].pmts[k].prop_est['delay'] = pars[rr + qq + pp]
                    pp += 1

        # calculate chi^2:
        chi2_t = 0.
        n_term_t = 0
        try:
            for i in range(n_mpmt):
                for l in range(w_led):
                    r = i * w_led + l
                    if data_exists['led'].get(r, False):
                        for j in range(n_mpmt):
                            for k in range(m_pmt):
                                p = j * m_pmt + k
                                if data_exists['pmt'].get(p, False):
                                    wght = ws[i][l][j][k]
                                    if wght > 0.:
                                        dev = wdts[i][l][j][k]/wght \
                                              - wcd.mpmts[i].prop_est['clock_offset'] \
                                              + wcd.mpmts[j].prop_est['clock_offset'] \
                                              - wcd.mpmts[i].leds[l].prop_est['delay'] \
                                              - wcd.mpmts[j].pmts[k].prop_est['delay']
                                        n_term_t += 1
                                        chi2_t += wght * dev ** 2
                                        if return_chisqs:
                                            chisqs[TC_Data.index(i,l,j,k)] = wght * dev ** 2
                                            devs[TC_Data.index(i,l,j,k)] = dev
                                            if dev <-1 or dev >1:
                                                print('Path with large deviation: mPMT Tran ' + str(i) + ' LED ' +str(l) + ' mPMT Rec ' + str(j) + ' PMT ' + str(k) + ', dev = ' + str(dev))
                                                bad_list.append([i,l,j,k])
                                            # calculate the light travel distance
                                            led = wcd.mpmts[i].leds[l]
                                            place = led.get_placement(place_info)
                                            led_loc = place['location']
                                            pmt = wcd.mpmts[j].pmts[k]
                                            place = pmt.get_placement(place_info)
                                            pmt_loc = place['location']
                                            light_vec = np.subtract(pmt_loc, led_loc)
                                            dist = np.linalg.norm(light_vec)
                                            dists[TC_Data.index(i,l,j,k)] = dist
                                            
        except:
            pass

        chi2_e = 0.
        n_term_e = 0
        if e_apply:
            for i in range(n_mpmt):
                for l in range(w_led):
                    r = i * w_led + l
                    if data_exists['led'].get(r, False):
                        dev = wcd.mpmts[i].leds[l].prop_est['delay'] - e_mean
                        we = np.power(e_sigma, -2)
                        n_term_e += 1
                        chi2_e += we * dev ** 2

        chi2_a = 0.
        n_term_a = 0
        if a_apply:
            for j in range(n_mpmt):
                for k in range(m_pmt):
                    p = j * m_pmt + k
                    if data_exists['pmt'].get(p, False):
                        dev = wcd.mpmts[j].pmts[k].prop_est['delay'] - a_mean
                        wa = np.power(a_sigma, -2)
                        n_term_a += 1
                        chi2_a += wa * dev ** 2

        chi2_d = (wcd.mpmts[ref_mpmt].prop_est['clock_offset'] - ref_mpmt_delay) ** 2 / ref_mpmt_delay_sigma ** 2

        chi2 = chi2_t
        n_dof = n_term_t - len(b)

        if return_chisqs:
            return chi2, n_dof, chisqs, devs, dists, bad_list
        else:
            return chi2, n_dof, bad_list
