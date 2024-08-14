using CSV, DataFrames, Random, JLD2
using OceanBioME: CarbonChemistry

using OceanBioME.Models: teos10_density, teos10_polynomial_approximation
using OceanBioME.Models.CarbonChemistryModel: K0, K1, K2, KF
using Dates: now

DIC_name = "G2tco2"
Alk_name = "G2talk"
T_name = "G2temperature"
S_name = "G2salinity"
P_name = "G2pressure"
Si_name = "G2silicate"
PO_name = "G2phosphate"
pH_name = "G2phtsinsitutp"
depth_name = "G2depth"
cruise_name = "G2cruise"
fCO2_name = "G2fco2"
lon_name = "G2longitude"
lat_name = "G2latitude"

DIC_name, Alk_name, T_name, S_name, P_name, Si_name, PO_name, pH_name

# remove low quality data and non QCed
# and drop a weird point from cruise with a massive errors (only 2 points)
#=data = filter(row -> any([row[name*"f"] == 2 for name in [DIC_name, Alk_name, S_name]]) && 
                     !any([row[name] == -9999 for name in [DIC_name, Alk_name, T_name, S_name]]) &&
                     #row[cruise_name] != 5005 &&
                     #row[cruise_name] != 345 &&
                     row[Alk_name] > 1500 &&
                     row["G2phtsqc"] == 1,  data)
=#
# if we don't have Si and PO₄ the error gets a lot bigger but not in the surface values (as much)

density_function = teos10_density
# pH_error = 0.00022 ± 0.01223 -> mean is zero
# pCO₂_error = -3 ± 28.8 -> mean is also zero
# with teos polynomial appriximation error is almost identical in pCO2 but slightly bigger in pH

function convert_data(d)
    st = now()
    # get the glodap data - downloaded "merged" file from https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/
    data = filter(row -> any([row[name*"f"] == 2 for name in [DIC_name, Alk_name, S_name, pH_name]]) && 
                        !any([row[name] == -9999 for name in [DIC_name, Alk_name, T_name, S_name, P_name, Si_name, PO_name, pH_name]]) &&
                        row["G2expocode"] != "325020210316" &&
                        row["G2expocode"] != "33RO20071215" &&
                        row[cruise_name] != 270 &&
                        #row[Alk_name] > 1500 &&
                        row["G2phtsqc"] == 1,
                        d)

    save_object("output/glodap_cleaned_data.jld2", data)
    et = now()
    @info "Data formatted & saved. \n" * "time taken: " * string(et - st)
end
data = 0

if isfile("output/glodap_cleaned_data.jld2") == false
    data = CSV.read("glodap/GLODAP.csv", DataFrame)
    convert_data(data)
else 
    data = load_object("output/glodap_cleaned_data.jld2")
end

Nrows = size(data, 1)

function get_cleaned_data() #cleans data to only include data points that have all measurements
    filtered_data  = []
    first = true

    Threads.@threads for n in 1:Nrows
        valid = true
        T = data[n, T_name]
        S = data[n, S_name]
        P = data[n, P_name]
        depth = data[n, depth_name]

        if P == -9999
            valid = false 
        else 
            P = 0.1 * P # dbar to bar
        end

        lon = data[n, lon_name]
        lat = data[n, lat_name]

        density = density_function(T, S, P, lon, lat)

        DIC = data[n, DIC_name] * density * 1e-3
        Alk = data[n, Alk_name] * density * 1e-3

        silicate = data[n, Si_name]
        phosphate = data[n, PO_name]
        
        if silicate == -9999 || phosphate == -9999
            valid = false 
        else
            silicate = data[n, Si_name] * density * 1e-3
            phosphate = data[n, PO_name] * density * 1e-3
        end
        
        if data[n, pH_name*"f"] == 2 && data[n, pH_name] != -9999 && data[n, fCO2_name*"f"] == 2 && data[n, fCO2_name*"temp"] != -9999
            glodap_pH = data[n, pH_name]
            glodap_pCO₂ = data[n, fCO2_name]
            if glodap_pH >= 1.0 && glodap_pCO₂ >= 50 && valid == true
                push!(filtered_data, 
                    NamedTuple{(:values, :measurements)}((
                            (lat = lat, 
                            lon = lon, 
                            depth = depth,
                            T = T, 
                            S = S, 
                            DIC = DIC,
                            Alk = Alk, 
                            P = P, 
                            silicate = silicate, 
                            phosphate = phosphate),
                            (pH = glodap_pH, 
                            pCO₂ = glodap_pCO₂))))
                #=if first == true 
                    println(NamedTuple{(:loc, :values, :measurements)}((
                        (lat = lat, lon = lon),
                        (T = T, S = S, P = P, DIC = DIC, Alk = Alk, silicate = silicate, phosphate = phosphate),
                        (pH = glodap_pH, pCO₂ = glodap_pCO₂))))
                    println(n)
                    first = false
                end =#
            end
        end
    end
    return filtered_data
end
