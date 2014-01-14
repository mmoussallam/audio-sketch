'''
manu_sandbox.mp_routines  -  Created on Oct 7, 2013
@author: M. Moussallam
'''
def _single_mp_run(orig_signal,
                   dictionary,
                   bound,
                   max_it_num,
                   debug=0,
                   pad=True,
                   clean=False,
                   silent_fail=False,
                   debug_iteration=-1,
                   unpad=False,
                   max_thread_num=None):
    """

    A single MP run: given a signal

    """
    
    # optional add zeroes to the edge
    if pad:
        orig_signal.pad(dictionary.get_pad())
    res_signal = orig_signal.copy()

    # FFTW Optimization for C code: initialize module global variables
    mp._initialize_fftw(dictionary, max_thread_num)

    # initialize blocks
    dictionary.initialize(res_signal)

    # initialize approximant
    current_approx = approx.Approx(
        dictionary, [], orig_signal, debug_level=debug)

    # residualEnergy
    res_energy = []

    it_number = 0
    current_srr = current_approx.compute_srr()
    current_lambda = 1
    # check if signal has null energy
    if res_signal.energy == 0:
        raise ValueError(" Null signal energy ")
        
    res_energy.append(res_signal.energy)

    # Decomposition loop: stopping criteria is either SNR or iteration number
    while (current_lambda > bound) & (it_number < max_it_num):

        # Compute inner products and selects the best atom
        dictionary.update(res_signal, it_number)

        # retrieve the best correlated atom
        best_atom = dictionary.get_best_atom(debug)

        if best_atom is None:
            print 'No atom selected anymore'
            return current_approx, res_energy


        if debug > 0:            
            mp._itprint_(it_number, best_atom)

        try:
            res_signal.subtract(best_atom, debug)
            dictionary.compute_touched_zone(best_atom)
        except ValueError:
#            if not silent_fail:
            print "Something wrong happened at iteration %d \
                     atom substraction abandonned"%it_number

            print "Atom Selected: ", best_atom
            return current_approx, res_energy

        if debug > 1:
            print "new residual energy of %2.2f"\
                % (np.sum(res_signal.data ** 2))

        if not unpad:
            res_energy.append(res_signal.energy)
        else:
            # only compute the energy without the padded borders where
            # eventually energy has been created
            padd = dictionary.get_pad()
            # assume padding is max dictionaty size
            res_energy.append(np.sum(res_signal.data[padd:-padd] ** 2))

        # add atom to dictionary
        current_approx.add(best_atom)

        # compute new SRR and increment iteration Number
        current_srr = current_approx.compute_srr(res_signal)
        current_lambda = np.sqrt(1 - res_energy[-1] / res_energy[-2])

        if (current_lambda <= bound):
            current_approx.remove(best_atom, position=-1)
            if (debug > 0):
                debug_str = "Lambda of %2.2f - bound of %2.2f stopping at: \
                            %d iterations srr of %2.4f" % (current_lambda ,
                                                           bound, it_number,
                                                           current_srr)
    
                print debug_str
            #            _Logger.debug(debug_str)
        if (debug > 1):
            print "SRR reached of %1.3f at iteration %d"%(current_srr,
                                                          it_number)                                       
               
        it_number += 1

        # cleaning for memory consumption control
        if clean:
            del best_atom.waveform

    # VERY IMPORTANT CLEANING STAGE!
    mp._clean_fftw()

    return current_approx, res_energy