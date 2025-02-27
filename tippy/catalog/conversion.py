




#%%


# Conversion table written by Hyeonho Choi (2021.07.20)

class Conversion:
        
    @classmethod
    def APASS_to_PANSTARRS1(APASS_catalog):
        '''
        parameters
        ----------
        {APASS catalog filepath}
        
        returns 
        -------
        Converted APASS catalog in PS1 magnitude  
        
        notes 
        -----
        Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
        -----
        '''
                    
        ra = APASS_catalog['ra']
        dec = APASS_catalog['dec']
        g = APASS_catalog['g_mag']
        r = APASS_catalog['r_mag']
        i = APASS_catalog['i_mag']
        
        e_g = APASS_catalog['e_g_mag']
        e_r = APASS_catalog['e_r_mag']
        e_i = APASS_catalog['e_i_mag']

        gr = g-r
        ri = r-i
        
        g_c = g - 0.009 - 0.061*gr
        r_c = r + 0.065 - 0.026*gr
        i_c = i - 0.015 - 0.068*ri
        
        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.026**2 + (0.061*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.027**2 + (0.026*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.045**2 + (0.068*e_gr)**2)

        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c
                }
        
        atable = pd.DataFrame(source)
        ctable = Table.from_pandas(atable)
        result = self.match_digit_tbl(ctable)
        return result

    @classmethod
    def APASS_to_JH(APASS_catalog):
        '''
        parameters
        ----------
        {APASS catalog filepath}
        
        returns 
        -------
        Converted APASS catalog in Johnson-Cousins magnitude  
        
        notes 
        -----
        Conversion equation : https://arxiv.org/pdf/astro-ph/0609121v1.pdf (Jordi(2006))
        More information about SDSS conversion : https://www.sdss.org/dr12/algorithms/sdssubvritransform/
        -----
        '''
        
        ra = APASS_catalog['ra']
        dec = APASS_catalog['dec']
        B = APASS_catalog['B_mag']
        V = APASS_catalog['V_mag']
        g = APASS_catalog['g_mag']
        r = APASS_catalog['r_mag']
        i = APASS_catalog['i_mag']

        e_B = APASS_catalog['e_B_mag']
        e_V = APASS_catalog['e_V_mag']
        e_g = APASS_catalog['e_g_mag']
        e_r = APASS_catalog['e_r_mag']
        e_i = APASS_catalog['e_i_mag']

        ri = r-i

        e_ri = np.sqrt(e_r**2+e_i**2)
        
        R = r - 0.153*ri - 0.117
        I = R -0.930*ri - 0.259

        e_R = np.sqrt(e_r**2+ 0.003**2 + (0.153*e_ri)**2)
        e_I = np.sqrt(e_r**2+ 0.002**2 + (0.930*e_ri)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'B_mag':B,
                'e_B_mag':e_B,
                'V_mag':V,
                'e_V_mag':e_V,
                'R_mag':R,
                'e_R_mag':e_R,
                'I_mag':I,
                'e_I_mag':e_I
                }
        
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result
    
    @classmethod
    def PANSTARRS1_to_SDSS(PANSTARR_catalog):
        '''
        parameters
        ----------
        {PanSTARRS DR1 catalog filepath}
        
        returns 
        -------
        Converted PanSTARRS catalog in SDSS magnitude  
        
        notes 
        -----
        Conversion equation : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
        -----
        '''
                    
        ra = PANSTARR_catalog['ra']
        dec = PANSTARR_catalog['dec']
        g = PANSTARR_catalog['g_mag']
        r = PANSTARR_catalog['r_mag']
        i = PANSTARR_catalog['i_mag']
        
        e_g = PANSTARR_catalog['e_g_mag']
        e_r = PANSTARR_catalog['e_r_mag']
        e_i = PANSTARR_catalog['e_i_mag']
        
        gr = g-r
        ri = r-i
        
        g_c = g + 0.014 + 0.162*gr
        r_c = r - 0.001 + 0.011*gr
        i_c = i - 0.004 + 0.020*gr
    
        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_ri = np.sqrt((e_r)**2+(e_i)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.009**2 + (0.162*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.004**2 + (0.011*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.005**2 + (0.020*e_gr)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c,
                }
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result

    @classmethod
    def PANSTARRS1_to_SDSS(PANSTARR_catalog):
        '''
        parameters
        ----------
        {PanSTARRS DR1 catalog filepath}
        
        returns 
        -------
        Converted PanSTARRS catalog in SDSS magnitude  
        
        notes 
        -----
        Conversion equation : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
        -----
        '''
                    
        ra = PANSTARR_catalog['ra']
        dec = PANSTARR_catalog['dec']
        g = PANSTARR_catalog['g_mag']
        r = PANSTARR_catalog['r_mag']
        i = PANSTARR_catalog['i_mag']
        z = PANSTARR_catalog['z_mag']

        gk = PANSTARR_catalog['g_Kmag']
        rk = PANSTARR_catalog['r_Kmag']
        ik = PANSTARR_catalog['i_Kmag']
        zk = PANSTARR_catalog['z_Kmag']
        
        e_g = PANSTARR_catalog['e_g_mag']
        e_r = PANSTARR_catalog['e_r_mag']
        e_i = PANSTARR_catalog['e_i_mag']
        e_z = PANSTARR_catalog['e_z_mag']
        
        e_gk = PANSTARR_catalog['e_g_Kmag']
        e_rk = PANSTARR_catalog['e_r_Kmag']
        e_ik = PANSTARR_catalog['e_i_Kmag']
        e_zk = PANSTARR_catalog['e_z_Kmag']
        
        gr = g-r
        grk = gk-rk
        ri = r-i
        rik = rk-ik
        
        g_c = g + 0.014 + 0.162*gr
        r_c = r - 0.001 + 0.011*gr
        i_c = i - 0.004 + 0.020*gr
        z_c = z + 0.013 - 0.050*gr
        
        gk_c = gk + 0.014 + 0.162*grk
        rk_c = rk - 0.001 + 0.011*grk
        ik_c = ik - 0.004 + 0.020*grk
        zk_c = zk + 0.013 - 0.050*grk

        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_grk = np.sqrt((e_gk)**2+(e_rk)**2)
        e_ri = np.sqrt((e_r)**2+(e_i)**2)
        e_rik = np.sqrt((e_rk)**2+(e_ik)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.009**2 + (0.162*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.004**2 + (0.011*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.005**2 + (0.020*e_gr)**2)
        e_z_c = np.sqrt((e_z)**2 + 0.010**2 + (0.050*e_gr)**2)
        
        e_gk_c = np.sqrt((e_gk)**2 + 0.009**2 + (0.162*e_grk)**2)
        e_rk_c = np.sqrt((e_rk)**2 + 0.004**2 + (0.011*e_grk)**2)
        e_ik_c = np.sqrt((e_ik)**2 + 0.005**2 + (0.020*e_grk)**2)
        e_zk_c = np.sqrt((e_zk)**2 + 0.010**2 + (0.050*e_grk)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c,
                'z_mag':z_c,
                'e_z_mag':e_z_c,
                'g_Kmag':gk_c,
                'e_g_Kmag':e_gk_c,
                'r_Kmag':rk_c,
                'e_r_Kmag':e_rk_c,
                'i_Kmag':ik_c,
                'e_i_Kmag':e_ik_c,
                'z_Kmag':zk_c,
                'e_z_Kmag':e_zk_c}
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result

    @classmethod
    def PANSTARRS1_to_APASS(PANSTARR_catalog):
        '''
        parameters
        ----------
        {PanSTARRS DR1 catalog filepath}
        
        returns 
        -------
        Converted PanSTARRS catalog in APASS magnitude  
        
        notes 
        -----
        Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
        -----
        '''
                    
        ra = PANSTARR_catalog['ra']
        dec = PANSTARR_catalog['dec']
        g = PANSTARR_catalog['g_mag']
        r = PANSTARR_catalog['r_mag']
        i = PANSTARR_catalog['i_mag']

        gk = PANSTARR_catalog['g_Kmag']
        rk = PANSTARR_catalog['r_Kmag']
        ik = PANSTARR_catalog['i_Kmag']
        
        e_g = PANSTARR_catalog['e_g_mag']
        e_r = PANSTARR_catalog['e_r_mag']
        e_i = PANSTARR_catalog['e_i_mag']
        
        e_gk = PANSTARR_catalog['e_g_Kmag']
        e_rk = PANSTARR_catalog['e_r_Kmag']
        e_ik = PANSTARR_catalog['e_i_Kmag']
        
        gr = g-r
        grk = gk-rk
        
        g_c = g + 0.023 + 0.054*gr
        r_c = r - 0.058 + 0.023*gr
        i_c = i + 0.003 + 0.057*gr

        gk_c = gk + 0.023 + 0.054*grk
        rk_c = rk - 0.058 + 0.023*grk
        ik_c = ik + 0.003 + 0.057*grk
        
        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_grk = np.sqrt((e_gk)**2+(e_rk)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.032**2 + (0.054*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.039**2 + (0.023*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.050**2 + (0.057*e_gr)**2)
        
        e_gk_c = np.sqrt((e_gk)**2 + 0.032**2 + (0.054*e_grk)**2)
        e_rk_c = np.sqrt((e_rk)**2 + 0.039**2 + (0.023*e_grk)**2)
        e_ik_c = np.sqrt((e_ik)**2 + 0.050**2 + (0.057*e_grk)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c,
                'g_Kmag':gk_c,
                'e_g_Kmag':e_gk_c,
                'r_Kmag':rk_c,
                'e_r_Kmag':e_rk_c,
                'i_Kmag':ik_c,
                'e_i_Kmag':e_ik_c}
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result
    
    @classmethod
    def PANSTARRS1_to_SMSS(PANSTARR_catalog):
        '''
        parameters
        ----------
        {PanSTARRS DR1 catalog filepath}
        
        returns 
        -------
        Converted PanSTARRS catalog in Skymapper DR1 magnitude  
        
        notes 
        -----
        Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
        -----
        '''
                    
        ra = PANSTARR_catalog['ra']
        dec = PANSTARR_catalog['dec']
        g = PANSTARR_catalog['g_mag']
        r = PANSTARR_catalog['r_mag']
        i = PANSTARR_catalog['i_mag']
        z = PANSTARR_catalog['z_mag']

        gk = PANSTARR_catalog['g_Kmag']
        rk = PANSTARR_catalog['r_Kmag']
        ik = PANSTARR_catalog['i_Kmag']
        zk = PANSTARR_catalog['z_Kmag']
        
        e_g = PANSTARR_catalog['e_g_mag']
        e_r = PANSTARR_catalog['e_r_mag']
        e_i = PANSTARR_catalog['e_i_mag']
        e_z = PANSTARR_catalog['e_z_mag']
        
        e_gk = PANSTARR_catalog['e_g_Kmag']
        e_rk = PANSTARR_catalog['e_r_Kmag']
        e_ik = PANSTARR_catalog['e_i_Kmag']
        e_zk = PANSTARR_catalog['e_z_Kmag']
        
        gr = g-r
        grk = gk-rk
        ri = r-i
        rik = rk-ik
        
        g_c = g + 0.010 - 0.228*gr
        r_c = r + 0.004 + 0.039*gr
        i_c = i + 0.008 - 0.110*ri
        z_c = z - 0.004 - 0.097*ri

        gk_c = gk + 0.010 - 0.228*grk
        rk_c = rk + 0.004 + 0.039*grk
        ik_c = ik + 0.008 - 0.110*rik
        zk_c = zk - 0.004 - 0.097*rik
        
        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_grk = np.sqrt((e_gk)**2+(e_rk)**2)
        e_ri = np.sqrt((e_r)**2+(e_i)**2)
        e_rik = np.sqrt((e_rk)**2+(e_ik)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.032**2 + (0.228*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.016**2 + (0.039*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.022**2 + (0.110*e_gr)**2)
        e_z_c = np.sqrt((e_z)**2 + 0.020**2 + (0.097*e_gr)**2)
        
        e_gk_c = np.sqrt((e_gk)**2 + 0.032**2 + (0.228*e_grk)**2)
        e_rk_c = np.sqrt((e_rk)**2 + 0.016**2 + (0.039*e_grk)**2)
        e_ik_c = np.sqrt((e_ik)**2 + 0.022**2 + (0.110*e_grk)**2)
        e_zk_c = np.sqrt((e_zk)**2 + 0.020**2 + (0.097*e_grk)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c,
                'z_mag':z_c,
                'e_z_mag':e_z_c,
                'g_Kmag':gk_c,
                'e_g_Kmag':e_gk_c,
                'r_Kmag':rk_c,
                'e_r_Kmag':e_rk_c,
                'i_Kmag':ik_c,
                'e_i_Kmag':e_ik_c,
                'z_Kmag':zk_c,
                'e_z_Kmag':e_zk_c}
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result

    @classmethod
    def PANSTARRS1_to_JH(PANSTARR_catalog):
        '''
        parameters
        ----------
        {PanSTARRS DR1 catalog filepath}
        
        returns 
        -------
        Converted PanSTARRS catalog in APASS magnitude  
        
        notes 
        -----
        Conversion equation : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
        -----
        '''
                    
        ra = PANSTARR_catalog['ra']
        dec = PANSTARR_catalog['dec']
        g = PANSTARR_catalog['g_mag']
        r = PANSTARR_catalog['r_mag']
        i = PANSTARR_catalog['i_mag']

        gk = PANSTARR_catalog['g_Kmag']
        rk = PANSTARR_catalog['r_Kmag']
        ik = PANSTARR_catalog['i_Kmag']
        
        e_g = PANSTARR_catalog['e_g_mag']
        e_r = PANSTARR_catalog['e_r_mag']
        e_i = PANSTARR_catalog['e_i_mag']
        
        e_gk = PANSTARR_catalog['e_g_Kmag']
        e_rk = PANSTARR_catalog['e_r_Kmag']
        e_ik = PANSTARR_catalog['e_i_Kmag']

        gr = g-r
        grk = gk-rk

        B = g + 0.213 + 0.587*gr
        V = r + 0.006 + 0.474*gr
        R = r - 0.138 - 0.131*gr
        I = i - 0.367 - 0.149*gr
        
        Bk = gk + 0.213 + 0.587*grk
        Vk = rk + 0.006 + 0.474*grk
        Rk = rk - 0.138 - 0.131*grk
        Ik = ik - 0.367 - 0.149*grk   

        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_grk = np.sqrt((e_g)**2)
        
        e_B_c = np.sqrt((e_g)**2 + 0.034**2 + (0.587*e_gr)**2)
        e_V_c = np.sqrt((e_r)**2 + 0.012**2 + (0.006*e_gr)**2)
        e_R_c = np.sqrt((e_r)**2 + 0.015**2 + (0.131*e_gr)**2)
        e_I_c = np.sqrt((e_i)**2 + 0.016**2 + (0.149*e_gr)**2)
        
        e_Bk_c = np.sqrt((e_gk)**2 + 0.034**2 + (0.587*e_grk)**2)
        e_Vk_c = np.sqrt((e_rk)**2 + 0.012**2 + (0.006*e_grk)**2)
        e_Rk_c = np.sqrt((e_rk)**2 + 0.015**2 + (0.131*e_grk)**2)
        e_Ik_c = np.sqrt((e_ik)**2 + 0.016**2 + (0.149*e_grk)**2)

        source = {'ra':ra,
                'dec':dec,
                'B_mag':B,
                'e_B_mag':e_B_c,
                'V_mag':V,
                'e_V_mag':e_V_c,
                'R_mag':R,
                'e_R_mag':e_R_c,
                'I_mag':I,
                'e_I_mag':e_I_c,
                'B_Kmag':Bk,
                'e_B_Kmag':e_Bk_c,
                'V_Kmag':Vk,
                'e_V_Kmag':e_Vk_c,
                'R_Kmag':Rk,
                'e_R_Kmag':e_Rk_c,
                'I_Kmag':Ik,
                'e_I_Kmag':e_Ik_c}
        
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result
    
    @classmethod
    def SMSS_to_PanSTARRS1(SMSS_catalog):
        '''
        parameters
        ----------
        {SMSS catalog filepath}
        
        returns 
        -------
        Converted SMSS catalog in PS1 magnitude  
        
        notes 
        -----
        Conversion equation : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
        -----
        '''
                    
        ra = SMSS_catalog['ra']
        dec = SMSS_catalog['dec']
        g = SMSS_catalog['g_mag']
        r = SMSS_catalog['r_mag']
        i = SMSS_catalog['i_mag']
        z = SMSS_catalog['z_mag']
        flag = SMSS_catalog['flag']
        ngood = SMSS_catalog['ngood']
        class_star = SMSS_catalog['class_star']
        
        e_g = SMSS_catalog['e_g_mag']
        e_r = SMSS_catalog['e_r_mag']
        e_i = SMSS_catalog['e_i_mag']
        e_z = SMSS_catalog['e_z_mag']

        gr = g-r
        ri = r-i
        
        g_c = g + 0.004 + 0.272*gr
        r_c = r - 0.016 - 0.035*gr
        i_c = i - 0.011 + 0.100*ri
        z_c = z + 0.009 + 0.082*ri

        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_ri = np.sqrt((e_r)**2+(e_i)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.029**2 + (0.272*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.021**2 + (0.035*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.016**2 + (0.100*e_ri)**2)
        e_z_c = np.sqrt((e_i)**2 + 0.020**2 + (0.082*e_ri)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c,
                'z_mag':z_c,
                'e_z_mag':e_z_c,
                'flag':flag,
                'ngood':ngood,
                'class_star':class_star
                }
        
        atable = pd.DataFrame(source)
        ctable = Table.from_pandas(atable)
        result = self.match_digit_tbl(ctable)
        return result

    @classmethod
    def SMSS_to_SDSS(SMSS_catalog):
        '''
        parameters
        ----------
        {SMSS catalog filepath}
        
        returns 
        -------
        Converted SMSS catalog in SDSS magnitude  
        
        notes 
        -----
        This conversion is performed by two conversion equation (SMSS > PS1 > SDSS)
        Conversion equation(SMSS1>PS1) : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
        Conversion equation(PS1>SDSS) : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
        -----
        '''
        
        pcatalog = SMSS_to_PanSTARRS1(SMSS_catalog)
        
        flag = pcatalog['flag']
        ngood = pcatalog['ngood']
        class_star = pcatalog['class_star']
        
        ra = pcatalog['ra']
        dec = pcatalog['dec']
        g = pcatalog['g_mag']
        r = pcatalog['r_mag']
        i = pcatalog['i_mag']
        z = pcatalog['z_mag']


        e_g = pcatalog['e_g_mag']
        e_r = pcatalog['e_r_mag']
        e_i = pcatalog['e_i_mag']
        e_z = pcatalog['e_z_mag']
        
        gr = g-r
        ri = r-i
        
        g_c = g + 0.014 + 0.162*gr
        r_c = r - 0.001 + 0.011*gr
        i_c = i - 0.004 + 0.020*gr
        z_c = z + 0.013 - 0.050*gr

        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        e_ri = np.sqrt((e_r)**2+(e_i)**2)
        
        e_g_c = np.sqrt((e_g)**2 + 0.009**2 + (0.162*e_gr)**2)
        e_r_c = np.sqrt((e_r)**2 + 0.004**2 + (0.011*e_gr)**2)
        e_i_c = np.sqrt((e_i)**2 + 0.005**2 + (0.020*e_gr)**2)
        e_z_c = np.sqrt((e_z)**2 + 0.010**2 + (0.050*e_gr)**2)
        
        source = {'ra':ra,
                'dec':dec,
                'g_mag':g_c,
                'e_g_mag':e_g_c,
                'r_mag':r_c,
                'e_r_mag':e_r_c,
                'i_mag':i_c,
                'e_i_mag':e_i_c,
                'z_mag':z_c,
                'e_z_mag':e_z_c,
                'flag':flag,
                'ngood':ngood,
                'class_star':class_star
                }
        
        atable = pd.DataFrame(source)
        ctable = Table.from_pandas(atable)
        result = self.match_digit_tbl(ctable)
        return result

    @classmethod
    def SMSS_to_JH(SMSS_catalog):
        '''
        parameters
        ----------
        {SMSS catalog filepath}
        
        returns 
        -------
        Converted SMSS catalog in Johnson-Cousins magnitude  
        
        notes 
        -----
        This conversion is performed by two conversion equation (SMSS > PS1 > JH)
        Conversion equation(SMSS1>PS1) : https://arxiv.org/pdf/1809.09157.pdf (Torny(2018))
        Conversion equation(PS1>JH) : https://iopscience.iop.org/article/10.1088/0004-637X/750/2/99/pdf (Torny(2012))
        -----
        '''
        
        pcatalog = SMSS_to_PanSTARRS1(SMSS_catalog)
        
        flag = pcatalog['flag']
        ngood = pcatalog['ngood']
        class_star = pcatalog['class_star']
        
        ra = pcatalog['ra']
        dec = pcatalog['dec']
        g = pcatalog['g_mag']
        r = pcatalog['r_mag']
        i = pcatalog['i_mag']
        
        e_g = pcatalog['e_g_mag']
        e_r = pcatalog['e_r_mag']
        e_i = pcatalog['e_i_mag']

        gr = g-r

        B = g + 0.213 + 0.587*gr
        V = r + 0.006 + 0.474*gr
        R = r - 0.138 - 0.131*gr
        I = i - 0.367 - 0.149*gr

        e_gr = np.sqrt((e_g)**2+(e_r)**2)
        
        e_B_c = np.sqrt((e_g)**2 + 0.034**2 + (0.587*e_gr)**2)
        e_V_c = np.sqrt((e_r)**2 + 0.012**2 + (0.006*e_gr)**2)
        e_R_c = np.sqrt((e_r)**2 + 0.015**2 + (0.131*e_gr)**2)
        e_I_c = np.sqrt((e_i)**2 + 0.016**2 + (0.149*e_gr)**2)

        source = {'ra':ra,
                'dec':dec,
                'B_mag':B,
                'e_B_mag':e_B_c,
                'V_mag':V,
                'e_V_mag':e_V_c,
                'R_mag':R,
                'e_R_mag':e_R_c,
                'I_mag':I,
                'e_I_mag':e_I_c,
                'flag':flag,
                'ngood':ngood,
                'class_star':class_star
                }
        
        ptable = pd.DataFrame(source)
        ctable = Table.from_pandas(ptable)
        result = self.match_digit_tbl(ctable)
        return result