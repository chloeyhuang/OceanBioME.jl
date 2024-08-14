using CSV
using DataFrames
using DelimitedFiles
using OceanBioME: CarbonChemistry

# load the GLODAP data
gd=CSV.read("../data/GLODAP/GLODAP.csv",DataFrame,delim=',')

# drop rows with missing values in DIC, Alk, T, S, silicate or phosphate columns
gd_clean=dropmissing(gd,[:"DIC_[umol/kg]",:"Alk_[umol/kg]",:"Temperature_[degC]",:"Salinity",:"Silicate_[umol/kg]",:"Phosphate_[umol/kg]"])

# make vector for OceanBioME output
N=size(gd_clean,1)
OceanBioME_pCO2=vec(ones(N,1))

carbon_chemistry=CarbonChemistry()
for i=1:N
    OceanBioME_pCO2[i]=carbon_chemistry(gd_clean[i,"DIC_[umol/kg]"],gd_clean[i,"Alk_[umol/kg]"],gd_clean[i,"Temperature_[degC]"],gd_clean[i,"Salinity"];
        silicate = gd_clean[i,"Silicate_[umol/kg]"],
        phosphate = gd_clean[i,"Phosphate_[umol/kg]"])
end

# append OceanBioME output to data frame
gd_clean[!,"OceanBioME_pCO2_[uatm]"]=OceanBioME_pCO2

# save data frame
CSV.write("../data/GLODAP/GLODAP_OceanBioME_20240719.csv",gd_clean)