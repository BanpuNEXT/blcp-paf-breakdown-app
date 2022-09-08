# BLCP anomaly detection
This is an ML application to detect abnormal pattern of indicators using AutoEncoder models to detect Praimary Air Fan breakdown.

## Installation
### Install python3.9
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.9
```
### Install required libraries
```
pip3 install -r requirements.txt
```

## Run service
```
cd src/
export FLASK_APP=app
export FLASK_ENV=prod
flask run
```

## Latest breakdown status of all devices.
```
/predict/paf/status
```
### Input
None

### Output
Return latest breakdown status of all devices and features.
```json
{
    "data": {
        "status": [
            {
                "deviceId": "ObjectId",
                "feature": "string",
                "status_message": "string"
            },
            {
                "deviceId": "ObjectId",
                "feature": "string",
                "status_message": "string"
            }
        ]
    }
}
```
### Visualisation
Print status_message of each deviceId and feature.

Example plot in in /sample_output/status.

## Error ratio of a feature of a device.
```
/predict/paf/error_ratio
```
### Input
| KEY       | VALUE     |
| --------- |:---------:|
| deviceId    | String    |
| feature | String   |

### Output
Return series of raw input and error ratio along with config values.
```json
{
    "data": {
        "error_ratio": [
            {
                "publishTimestamp": "ISODate",
                "raw_data": "float",
                "error_ratio": "float"
            },
            {
                "publishTimestamp": "ISODate",
                "raw_data": "float",
                "error_ratio": "float"
            },        
        ],
        "config_value": [
            {
                "error_ratio_threshold": "float"
            }
        ]
    }
}
```
### Visualisation
Print status_message of each deviceId and feature.
 - Plot series of raw_data and publishTimestamp.
 - Plot series of error_ratio and publishTimestamp with error_ratio_threshold as baseline horizontal.

Example plot in /sample_output/error_ratio.