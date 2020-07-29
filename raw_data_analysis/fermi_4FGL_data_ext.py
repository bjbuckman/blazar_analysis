from astropy.io import fits
from astroquery.ned import Ned
import pandas as pd
import numpy as np
import time

all_fermi = fits.open('gll_psc_v19.fit')

## all_fermi[1] has all info

num_objects = len(all_fermi[1].data)

name_4FGL = []

s_type = []
associated_source = []
associated_ra     = []
associated_de     = []

flux = []
flux_err = []
flux_E = []
flux_E_err = []

PL_index = []
unc_PL_index = []

variability_index = [] 
frac_variability = []
unc_frac_variability = []

TeVCat_Flag = []
alt_name_gamma =[]

redshift = []
redshift_flag = []

median_optical_flux = []
num_optical_flux = []

for ii in range(0,num_objects):
	source_type = all_fermi[1].data[ii][63]
	# if source_type == 'agn' or source_type == 'bcu' or source_type == 'bll' or source_type == 'css' or source_type == 'fsrq' or source_type == 'ssrq':
	if source_type == 'bll' or source_type == 'BLL' or source_type == 'fsrq' or source_type == 'FSRQ' or source_type == 'bcu' or source_type == 'BCU':
	
		name_4FGL+= [all_fermi[1].data[ii][0]]
		s_type   += [source_type]
		associated_source += [all_fermi[1].data[ii][65]]
		associated_ra     += [all_fermi[1].data[ii][69]]
		associated_de     += [all_fermi[1].data[ii][70]]
		
		flux += [all_fermi[1].data[ii][15]]
		flux_err += [all_fermi[1].data[ii][16]]
		flux_E += [all_fermi[1].data[ii][17]*6.24e5]
		flux_E_err += [all_fermi[1].data[ii][18]*6.24e5]
		
		PL_index += [all_fermi[1].data['PL_Index'][ii]]
		unc_PL_index += [all_fermi[1].data['Unc_PL_Index'][ii]]
		
		variability_index += [all_fermi[1].data['Variability_Index'][ii]]
		frac_variability += [all_fermi[1].data['Frac_Variability'][ii]]
		unc_frac_variability += [all_fermi[1].data['Unc_Frac_Variability'][ii]]
		TeVCat_Flag += [all_fermi[1].data['TeVCat_Flag'][ii]]
		alt_name_gamma += [all_fermi[1].data['ASSOC_FGL'][ii]]
		
		print(all_fermi[1].data[ii][0], all_fermi[1].data[ii][65])
		try:
			result = Ned.query_object(all_fermi[1].data[ii][65])
			redshift_temp = result['Redshift'][0]
			redshift_flag_temp = result['Redshift Flag'][0]
		except:
			redshift_temp = '--'
			redshift_flag_temp = ''
		redshift+= [redshift_temp]
		redshift_flag+= [redshift_flag_temp]
			
		try:
			result1 = Ned.get_table(all_fermi[1].data[ii][65])
			# optical_fluxes = result1[['Observed Passband', 'Frequency', 'Photometry Measurement', 'Uncertainty', 'Units', 'Flux Density', 'Lower limit of uncertainty']][np.logical_and(result1['Frequency'] < 1e16, result1['Frequency'] > 1e14)] 
			optical_fluxes = result1['Flux Density'][np.logical_and(result1['Frequency'] < 1e16, result1['Frequency'] > 1e14)]
			optical_fluxes = np.array(optical_fluxes)
			num_optical_flux_temp = len(optical_fluxes)
			median_optical_flux_temp = np.nanmedian(optical_fluxes)*1.e3
		except:
			median_optical_flux_temp = '--' 
			num_optical_flux_temp = '--'
		median_optical_flux+= [median_optical_flux_temp]
		num_optical_flux+= [num_optical_flux_temp]
		
		# print(all_fermi[1].data[ii][0])
		time.sleep(1)


output_dict = {'associated_de':associated_de, 'associated_ra':associated_ra, 'associated_source':associated_source, 'source_type':s_type, 'name_4FGL':name_4FGL, 'alt_name_gamma':alt_name_gamma, 'flux':flux, 'flux_err':flux_err, 'flux_E':flux_E, 'flux_E_err':flux_E_err, 'redshift':redshift, 'redshift_flag':redshift_flag, 'median_optical_flux':median_optical_flux, 'num_optical_flux':num_optical_flux, 'PL_index':PL_index, 'PL_index_err':unc_PL_index, 'variability_index':variability_index, 'frac_variability':frac_variability, 'frac_variability_err':unc_frac_variability, 'TeVCat_Flag':TeVCat_Flag}

# print(len(associated_de), len(redshift), len(redshift_flag), len(median_optical_flux), len(num_optical_flux))

output_df = pd.DataFrame(data=output_dict)

cols = ['name_4FGL', 'alt_name_gamma', 'source_type', 'associated_source', 'associated_ra', 'associated_de', 'flux', 'flux_err', 'flux_E', 'flux_E_err', 'redshift', 'redshift_flag', 'median_optical_flux', 'num_optical_flux', 'PL_index', 'PL_index_err', 'variability_index', 'frac_variability', 'frac_variability_err', 'TeVCat_Flag']

output_df = output_df[cols]

# output_df = output_df.sort_values(by=['flux'], ascending=False)
output_df = output_df.reset_index(drop=True)

output_df.to_csv('fermi_4FGL_associations_ext_4FGLorder.csv')

output_df = output_df.sort_values(by=['flux'], ascending=False)
output_df = output_df.reset_index(drop=True)
output_df.to_csv('fermi_4FGL_associations_ext_GRPHorder.csv')

print(output_df)

# result_table = Ned.query_object(associated_source[0])
# print(result_table[0][1])

# co = coordinates.SkyCoord(ra=fermi_RA[0], dec=fermi_DE[0], unit=(u.deg, u.deg), frame='icrs')
# result_table = Ned.query_region(co, radius=radius[0]*u.deg, equinox='J2000.0')
# print(result_table)
# print(result_table[0][0])

# response = requests.post("https://ned.ipac.caltech.edu/tap/sync?query=SELECT+*+FROM+objdir+WHERE+CONTAINS(POINT('J2000',ra,dec),CIRCLE('J2000',"+str(fermi_RA[0])+","+str(fermi_DE[0])+","+str(radius[0])+"))=1&LANG=ADQL&REQUEST=doQuery&FORMAT=text")

# print(response.text)

# curl command
# curl -o out.txt "https://ned.ipac.caltech.edu/tap/sync?query=SELECT+*+FROM+objdir+WHERE+CONTAINS(POINT('J2000',ra,dec),CIRCLE('J2000',66.76957,26.10453,0.01))=1&LANG=ADQL&REQUEST=doQuery&FORMAT=text" 