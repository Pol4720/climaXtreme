"""
Data cleaning functions for climate data.
"""

import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, when, regexp_extract

logger = logging.getLogger(__name__)

def add_geographic_columns(df: DataFrame) -> DataFrame:
    """
    Adds region and continent columns based on coordinates and country.
    
    Args:
        df: Input DataFrame with latitude, longitude, and country columns.
        
    Returns:
        DataFrame with region and continent columns added.
    """
    # Extract numeric latitude (e.g., "57.05N" -> 57.05, "23.5S" -> -23.5)
    df = df.withColumn(
        "lat_numeric",
        when(
            col("latitude").rlike(".*N$"),
            regexp_extract(col("latitude"), r"([\d.]+)", 1).cast("double")
        ).otherwise(
            -regexp_extract(col("latitude"), r"([\d.]+)", 1).cast("double")
        )
    )
    
    # Extract numeric longitude (e.g., "10.2E" -> 10.2, "75.5W" -> -75.5)
    df = df.withColumn(
        "lon_numeric",
        when(
            col("longitude").rlike(".*E$"),
            regexp_extract(col("longitude"), r"([\d.]+)", 1).cast("double")
        ).otherwise(
            -regexp_extract(col("longitude"), r"([\d.]+)", 1).cast("double")
        )
    )
    
    # Add continent based on country/coordinates
    df = df.withColumn(
        "continent",
        # North America
        when(col("country").isin(
            "United States", "Canada", "Mexico", "Cuba", "Jamaica", "Haiti", 
            "Dominican Republic", "Guatemala", "Honduras", "Nicaragua", "Costa Rica", 
            "Panama", "El Salvador", "Puerto Rico", "Bahamas", "Belize", "Trinidad And Tobago"
        ), "North America")
        # South America
        .when(col("country").isin(
            "Brazil", "Argentina", "Chile", "Peru", "Colombia", "Venezuela", 
            "Ecuador", "Bolivia", "Paraguay", "Uruguay", "Guyana", "Suriname"
        ), "South America")
        # Asia
        .when(col("country").isin(
            "China", "Japan", "India", "Indonesia", "Philippines", "Vietnam", "Thailand", 
            "South Korea", "Malaysia", "Singapore", "Pakistan", "Bangladesh", "Myanmar",
            "Burma", "Taiwan", "Sri Lanka", "Nepal", "Afghanistan", "Iraq", "Iran", 
            "Saudi Arabia", "Turkey", "Israel", "Syria", "Jordan", "Lebanon", 
            "United Arab Emirates", "Kuwait", "Qatar", "Bahrain", "Oman", "Yemen", 
            "Kazakhstan", "Uzbekistan", "Mongolia", "Cambodia", "Laos", "Hong Kong",
            "Tajikistan", "Turkmenistan", "Kyrgyzstan", "North Korea"
        ), "Asia")
        # Europe
        .when(col("country").isin(
            "Germany", "France", "United Kingdom", "Italy", "Spain", "Poland", "Romania", 
            "Netherlands", "Belgium", "Greece", "Portugal", "Czech Republic", "Sweden", 
            "Hungary", "Austria", "Switzerland", "Bulgaria", "Denmark", "Finland", "Norway", 
            "Ireland", "Croatia", "Slovakia", "Slovenia", "Latvia", "Estonia", "Lithuania", 
            "Luxembourg", "Malta", "Cyprus", "Iceland", "Russia", "Ukraine", "Belarus",
            "Albania", "Montenegro", "Georgia", "Azerbaijan", "Armenia", "Macedonia",
            "Serbia", "Bosnia And Herzegovina", "Moldova", "Kosovo"
        ), "Europe")
        # Africa
        .when(col("country").isin(
            "Nigeria", "Egypt", "South Africa", "Algeria", "Morocco", "Sudan", "Ethiopia", 
            "Kenya", "Tanzania", "Uganda", "Ghana", "Mozambique", "Madagascar", "Cameroon", 
            "Angola", "Senegal", "Zimbabwe", "Tunisia", "Libya", "Zambia", "Rwanda", "Mali", 
            "Niger", "Burkina Faso", "Malawi", "Somalia", "Chad", "Togo", "Benin", "Eritrea",
            "Djibouti", "Lesotho", "Sierra Leone", "Burundi", "Gabon", "Mauritania", "Congo",
            "Liberia", "Namibia", "Guinea", "Gambia", "Botswana", "Equatorial Guinea", 
            "Reunion", "Central African Republic", "Swaziland", "Côte D'Ivoire", 
            "Cote D'Ivoire", "Guinea Bissau", "Mauritius", "Congo (Democratic Republic Of The)",
            "Cape Verde", "Comoros", "Seychelles", "Sao Tome And Principe"
        ), "Africa")
        # Oceania
        .when(col("country").isin(
            "Australia", "New Zealand", "Papua New Guinea", "Fiji", "Solomon Islands", 
            "Vanuatu", "Samoa", "Tonga", "Micronesia", "Kiribati", "Palau", "Marshall Islands",
            "Nauru", "Tuvalu"
        ), "Oceania")
        .otherwise("Other")
    )
    
    # Add region based on latitude bands
    df = df.withColumn(
        "region",
        when(col("lat_numeric") > 66.5, "Arctic")
        .when((col("lat_numeric") > 45) & (col("lat_numeric") <= 66.5), "Northern Temperate")
        .when((col("lat_numeric") > 23.5) & (col("lat_numeric") <= 45), "Northern Subtropical")
        .when((col("lat_numeric") > 0) & (col("lat_numeric") <= 23.5), "Northern Tropical")
        .when((col("lat_numeric") > -23.5) & (col("lat_numeric") <= 0), "Southern Tropical")
        .when((col("lat_numeric") > -45) & (col("lat_numeric") <= -23.5), "Southern Subtropical")
        .when((col("lat_numeric") > -66.5) & (col("lat_numeric") <= -45), "Southern Temperate")
        .otherwise("Antarctic")
    )
    
    return df

def clean_temperature_data(df: DataFrame) -> DataFrame:
    """
    Cleans temperature data by removing outliers and invalid values.
    
    Args:
        df: Input DataFrame with temperature data.
        
    Returns:
        A cleaned DataFrame.
    """
    cleaned_df = (
        df
        .filter(col("temperature").isNotNull())
        .filter(col("year").isNotNull() & (col("year") > 1750) & (col("year") <= 2030))
        .filter(col("month").isNotNull() & (col("month") >= 1) & (col("month") <= 12))
        .filter(col("temperature") >= -100.0)
        .filter(col("temperature") <= 60.0)
    )
    
    original_count = df.count()
    cleaned_count = cleaned_df.count()
    removed_count = original_count - cleaned_count
    
    if original_count > 0:
        logger.info(
            f"Data cleaning: {original_count} -> {cleaned_count} records "
            f"({removed_count} removed, {removed_count/original_count*100:.1f}%)"
        )
    
    return cleaned_df