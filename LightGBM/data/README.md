# Dataset format
Data Dictionary – Yellow TaxiTripRecords

This data dictionary describes yellow taxi trip data. For a dictionary describing green taxi data, or a map of the TLC Taxi Zones, please visit[https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

| **Field Name** | **Description** |
| --- | --- |
| **VendorID** | A code indicating the TPEP provider that provided the record. <br>**1= Creative Mobile Technologies, LLC; 2= VeriFone Inc.** |
| **tpep\_pickup\_datetime** | The date and time when the meter was engaged. |
| **tpep\_dropoff\_datetime** | The date and time when the meter was disengaged. |
| **Passenger\_count** | The number of passengers in the vehicle. <br>This is a driver-entered value. |
| **Trip\_distance** | The elapsed trip distance in miles reported by the taximeter. |
| **PULocationID** | TLC Taxi Zone in which the taximeter was engaged |
| **DOLocationID** | TLC Taxi Zone in which the taximeter was disengaged |
| **RateCodeID** | The final rate code in effect at the end of the trip.<br>**1= Standard rate <br>2=JFK <br>3=Newark <br>4=Nassau or Westchester <br>5=Negotiated fare <br>6=Group ride** |
| **Store\_and\_fwd\_flag** | This flag indicates whether the trip record was held in vehicle memory before sending to the vendor, aka &quot;store and forward,&quot; because the vehicle did not have a connection to the server.<br>**Y= store and forward trip <br> N= not a store and forward trip** |
| **Payment\_type** | A numeric code signifying how the passenger paid for the trip. <br>**1= Credit card<br> 2= Cash<br>3= No charge<br> 4= Dispute <br>5= Unknown<br> 6= Voided trip** |
| **Fare\_amount** | The time-and-distance fare calculated by the meter. |
| **Extra** | Miscellaneous extras and surcharges. Currently, this only includes the $0.50 and $1 rush hour and overnight charges. |
| **MTA\_tax** | $0.50 MTA tax that is automatically triggered based on the metered rate in use. |
| **Improvement\_surcharge** | $0.30 improvement surcharge assessed trips at the flag drop. The improvement surcharge began being levied in 2015. |
| **Tip\_amount** | Tip amount – This field is automatically populated for credit card tips. Cash tips are not included. |
| **Tolls\_amount** | Total amount of all tolls paid in trip. |
| **Total\_amount** | The total amount charged to passengers. Does not include cash tips. |