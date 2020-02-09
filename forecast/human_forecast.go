package forecast

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	dataframe "github.com/rocketlaunchr/dataframe-go"
)

// Interval is used to set the Interval Period.
type Interval int

// HfModel is a Model Interface that holds necessary
// computed values and results for a HumanForecast Algorithm
type HfModel struct {
	data           *dataframe.DataFrame
	testData       *dataframe.DataFrame
	fcastData      *dataframe.DataFrame
	smoothingLevel float64
	trendLevel     float64
	startDate      *time.Time
	endDate        *time.Time
	period         time.Duration
	interval       Interval
	alpha          float64
	beta           float64
	mae            float64
	sse            float64
	rmse           float64
	mape           float64
}

const (
	// See http://golang.org/pkg/time/#Parse
	timeFormat = "2006-01-02 15:00:00 +0000 UTC"

	// Hourly is Interval on hourly bases
	Hourly Interval = 0
	// Daily is Interval on Daily bases
	Daily Interval = 1
	// Weekly is Interval on weekly bases
	Weekly Interval = 2
)

// HumanForecastOptions contains options for HumanForecast Fit function.
type HumanForecastOptions struct {
	// Interval is used to set the Data Interval and calculate the period.
	Interval Interval
}

// HumanForecast Function receives a dataframe data with two columns
// One containig datetime data and the other containing floating values
// It returns a HfModel from which Fit and Predict method can be carried out.
func HumanForecast(d *dataframe.DataFrame) *HfModel {
	model := &HfModel{
		data:           &dataframe.DataFrame{},
		testData:       &dataframe.DataFrame{},
		fcastData:      &dataframe.DataFrame{},
		smoothingLevel: 0.0,
		trendLevel:     0.0,
		startDate:      nil,
		endDate:        nil,
		interval:       Daily,
		period:         0,
		alpha:          0.0,
		beta:           0.0,
		mae:            0.0,
		sse:            0.0,
		rmse:           0.0,
		mape:           0.0,
	}

	model.data = d
	return model
}

// Fit Method performs the splitting and training of the HfModel based on the Human Forecast algorithm.
// It returns a trained HfModel ready to carry out future predictions.
// The arguments α and β [0,1].
func (hf *HfModel) Fit(ctx context.Context, α, β float64, r dataframe.Range, options ...HumanForecastOptions) (*HfModel, error) {
	hf.data.Lock()
	defer hf.data.Unlock()

	var (
		period   time.Duration
		interval Interval
	)

	if len(options) > 0 {
		interval = options[0].Interval
	}

	hf.interval = interval

	start, end, err := r.Limits(hf.data.NRows(dataframe.DontLock))
	if err != nil {
		return nil, err
	}

	// Validation
	if end-start < 1 {
		return nil, errors.New("no values in series range")
	}

	if (α < 0.0) || (α > 1.0) {
		return nil, errors.New("α must be between [0,1]")
	}

	if (β < 0.0) || (β > 1.0) {
		return nil, errors.New("β must be between [0,1]")
	}

	hf.alpha = α
	hf.beta = β

	trainData := hf.data.Copy(dataframe.Range{Start: &start, End: &end})

	startDate, err := getDateTime(trainData.Series[0], start)
	if err != nil {
		return nil, err
	}

	endDate, err := getDateTime(trainData.Series[0], end)
	if err != nil {
		return nil, err
	}

	fullDateTimeDuration := endDate.Sub(*startDate)

	hf.startDate = startDate
	hf.endDate = endDate

	testStrt := end + 1
	testData := hf.data.Copy(dataframe.Range{Start: &testStrt})
	if testData.NRows(dataframe.DontLock) < 2 {
		return nil, errors.New("There should be a minimum of 2 data left as testing data")
	}
	hf.testData = testData

	var (
		nDfSplit           int // number of dataframe split
		splittedDataframes []*dataframe.DataFrame
	)
	// Split Dataframe according to interval specified
	switch interval {
	case Hourly:
		period = time.Hour
		nDfSplit = 24

		sampleDate := *startDate
		fmt.Println("Hour of Day", sampleDate.Hour())

		// sampleDate = sampleDate.Add(period)
		// nxtHour := sampleDate.Hour()
		// fmt.Println("Next Hour", nxtHour)
		for i := 0; i < nDfSplit; i++ {
			// TODO: make each iteration of i to run concurrently to improve speed

			date := dataframe.NewSeriesTime(fmt.Sprintf("HOUR: %d", sampleDate.Hour()), nil)
			value := dataframe.NewSeriesFloat64("Value", nil)
			newDataframe := dataframe.NewDataFrame(date, value)

			iterator := trainData.ValuesIterator(dataframe.ValuesOptions{InitialRow: 0, Step: 1, DontReadLock: true})
			trainData.Lock()
			for {
				if err := ctx.Err(); err != nil {
					return nil, err
				}

				row, vals, _ := iterator()
				if row == nil {
					break
				}

				currentDate, err := getDateTime(trainData.Series[0], *row)
				if err != nil {
					return nil, err
				}

				if currentDate.Hour() == sampleDate.Hour() {
					fmt.Println("current Date:", currentDate)

					insertVals := make([]interface{}, len(trainData.Series))
					for key, val := range vals {
						switch colName := key.(type) {
						case string:
							idx, _ := trainData.NameToColumn(colName)

							if idx > 0 && val == nil { // second column containing floating type values
								val, err = interpolateMissingData(trainData.Series[idx], *row, idx)
								if err != nil {
									return nil, err
								}
							}
							insertVals[idx] = val
						}
					}
					newDataframe.Append(&dataframe.DontLock, insertVals...)
				}
			}
			trainData.Unlock()

			sampleDate = sampleDate.Add(period)

			fmt.Println("Hour of Day:", sampleDate.Hour())

			splittedDataframes = append(splittedDataframes, newDataframe)
		}

	case Daily:
		// Split dataframe according to 7 days of the week
		period = time.Hour * 24
		nDfSplit = 7

		sampleDate := *startDate
		dayOfWeek := sampleDate.Weekday()
		fmt.Println("Day of week:", dayOfWeek)

		for i := 0; i < nDfSplit; i++ {
			// TODO: make each iteration of i to run concurrently to improve speed

			date := dataframe.NewSeriesTime(fmt.Sprintf("%s(s)", dayOfWeek), nil)
			value := dataframe.NewSeriesFloat64("Value", nil)
			newDataframe := dataframe.NewDataFrame(date, value)

			iterator := trainData.ValuesIterator(dataframe.ValuesOptions{0, 1, true})
			trainData.Lock()
			for {
				if err := ctx.Err(); err != nil {
					return nil, err
				}

				row, vals, _ := iterator()
				if row == nil {
					break
				}

				currentDate, err := getDateTime(trainData.Series[0], *row)
				if err != nil {
					return nil, err
				}

				if currentDate.Weekday() == dayOfWeek {
					fmt.Println("current Date:", currentDate)

					insertVals := make([]interface{}, len(trainData.Series))
					for key, val := range vals {
						switch colName := key.(type) {
						case string:
							idx, _ := trainData.NameToColumn(colName)

							if idx > 0 && val == nil { // second column containing floating type values
								val, err = interpolateMissingData(trainData.Series[idx], *row, idx)
								if err != nil {
									return nil, err
								}
							}
							insertVals[idx] = val
						}
					}
					newDataframe.Append(&dataframe.DontLock, insertVals...)
				}
			}
			trainData.Unlock()

			// Adjust sample date to point to next day
			sampleDate = sampleDate.Add(period)
			// set the new week day
			dayOfWeek = sampleDate.Weekday()
			fmt.Println("Day of week:", dayOfWeek)

			splittedDataframes = append(splittedDataframes, newDataframe)
		}

	case Weekly:
		// Split dataframe according to number of available weeks in dataset
		period = time.Hour * 24 * 7
		availableWeeks := math.Round(fullDateTimeDuration.Hours() / period.Hours())
		// For debuggin purpose, To be removed
		fmt.Println("full time duration:", fullDateTimeDuration)
		fmt.Println("Weeks:", availableWeeks)
		nDfSplit = int(availableWeeks)

		lowerBoundDate := *startDate
		upperBoundDate := lowerBoundDate.Add(period)

		// fmt.Println("lower Bound Date", lowerBoundDate)
		// fmt.Println("upper Bound Date:", upperBoundDate)

		// Split Dataframe
		for i := 0; i < nDfSplit; i++ {
			// TODO: make each iteration of i to run concurrently to improve speed

			date := dataframe.NewSeriesTime(fmt.Sprintf("week %d", i+1), nil)
			value := dataframe.NewSeriesFloat64("Value", nil)
			newDataframe := dataframe.NewDataFrame(date, value)

			iterator := trainData.ValuesIterator(dataframe.ValuesOptions{0, 1, true})
			trainData.Lock()
			for {
				if err := ctx.Err(); err != nil {
					return nil, err
				}

				row, vals, _ := iterator()
				if row == nil {
					break
				}

				currentDate, err := getDateTime(trainData.Series[0], *row)
				if err != nil {
					return nil, err
				}

				if (currentDate.Equal(lowerBoundDate) || currentDate.After(lowerBoundDate)) && currentDate.Before(upperBoundDate) {

					// fmt.Println("current Date:", currentDate)

					insertVals := make([]interface{}, len(trainData.Series))
					for key, val := range vals {
						switch colName := key.(type) {
						case string:
							idx, _ := trainData.NameToColumn(colName)

							if idx > 0 && val == nil { // second column containing floating type values
								val, err = interpolateMissingData(trainData.Series[idx], *row, idx)
								if err != nil {
									return nil, err
								}
							}
							insertVals[idx] = val
						}
					}

					newDataframe.Append(&dataframe.DontLock, insertVals...)
				}

			}
			trainData.Unlock()

			lowerBoundDate = upperBoundDate
			upperBoundDate = upperBoundDate.Add(period)
			splittedDataframes = append(splittedDataframes, newDataframe)

			// fmt.Println("lower Bound Date", lowerBoundDate)
			// fmt.Println("upper Bound Date:", upperBoundDate)
		}
	default:
		return nil, errors.New("unsupported time interval specified")
	}

	hf.period = period

	// Below printing statements for debug sake. To be removed...
	fmt.Println("period:", period)

	fmt.Println("Splitted Dataframes Dump...")
	// spew.Dump(splittedDataframes)
	for _, df := range splittedDataframes {
		fmt.Println(df.Table())
	}
	fmt.Println()

	return hf, nil
}
