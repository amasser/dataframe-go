package forecast

import (
	"context"
	"fmt"

	// "math/rand"
	"testing"
	// "time"

	dataframe "github.com/rocketlaunchr/dataframe-go"
)

func TestSes(t *testing.T) {
	ctx := context.Background()

	// data := dataframe.NewSeriesFloat64("Complete Data", nil, 445.43, 345.2, 565.56, 433.34, 585.23, 593.32, 641.43, 654.35, 234.65, 567.45, 645.45, 445.34, 564.65, 598.76, 676.54, 654.56, 564.76, 456.76, 656.57, 765.45, 755.43, 745.2, 665.56, 633.34, 585.23, 693.32, 741.43, 654.35, 734.65, 667.45, 545.45, 645.34, 754.65, 798.76, 776.54, 654.56, 664.76, 856.76, 776.57, 825.45, 815.43, 845.2, 765.56, 733.34, 785.23, 893.32, 841.43, 754.35, 524.65, 567.45, 715.45, 845.34, 864.65, 898.76, 876.54, 854.56, 864.76, 856.76, 726.57, 700.31, 815.43, 805.2, 855.56, 733.34, 785.23, 893.32, 641.43, 554.35, 734.63, 834.89)
	// m := 5
	alpha := 0.1

	data := dataframe.NewSeriesFloat64("simple data", nil, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	m := 10

	// fmt.Println(data.Table())

	fModel := SimpleExponentialSmoothing(data)

	fModelFit, err := fModel.Fit(ctx, alpha, dataframe.Range{End: &[]int{5}[0]})
	if err != nil {
		t.Errorf("unexpected error: %s\n", err)
	}
	//spew.Dump(fModelFit)

	fpredict, err := fModelFit.Predict(ctx, m)
	if err != nil {
		t.Errorf("unexpected error: %s\n", err)
	}

	fModelFit.Describe(ctx, MainData)

	fModelFit.Summary()

	fmt.Println(fpredict.Table())

}

func TestHw(t *testing.T) {
	ctx := context.Background()

	// 48 + 24 = 72 data pts + extra 12
	data := dataframe.NewSeriesFloat64("simple data", nil, 30, 21, 29, 31, 40, 48, 53, 47, 37, 39, 31, 29, 17, 9, 20, 24, 27, 35, 41, 38,
		27, 31, 27, 26, 21, 13, 21, 18, 33, 35, 40, 36, 22, 24, 21, 20, 17, 14, 17, 19,
		26, 29, 40, 31, 20, 24, 18, 26, 17, 9, 17, 21, 28, 32, 46, 33, 23, 28, 22, 27,
		18, 8, 17, 21, 31, 34, 44, 38, 31, 30, 26, 32, 45, 34, 30, 27, 25, 22, 28, 33, 42, 32, 40, 52)

	period := 12
	h := 24

	// fmt.Println(data.Table())

	fModel := HoltWinters(data)

	alpha := 0.45
	beta := 0.03
	gamma := 0.73

	fModelFit, err := fModel.Fit(ctx, alpha, beta, gamma, period, dataframe.Range{End: &[]int{71}[0]})
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}
	// spew.Dump(fModelFit)

	fpredict, err := fModelFit.Predict(ctx, h)
	if err != nil {
		t.Errorf("unexpected error: %s", err)
	}

	fModelFit.Describe(ctx, MainData)

	fModelFit.Summary()

	fmt.Println(fpredict.Table())
}

// func TestHumanForecast(t *testing.T) {
// 	ctx := context.Background()

// 	// Generating DateTime for 3weeks
// 	totalDays := 21 // days
// 	hours := 24
// 	days := make([]interface{}, totalDays*hours)
// 	i := 0
// 	for day := 0; day < totalDays; day++ {
// 		for hour := 0; hour < hours; hour++ {
// 			days[i] = time.Date(2019, time.October, 7+day, 0+hour, 0, 0, 0, time.UTC)
// 			i++
// 		}
// 	}
// 	// fmt.Println(days)

// 	dateTime := dataframe.NewSeriesTime("Hourly DateTime", nil, days...)
// 	// fmt.Println(dateTime.Table())

// 	// Generating 504 [21 * 24] random digits

// 	r := rand.New(rand.NewSource(50)) // setting a rand seed
// 	total := totalDays * hours
// 	amounts := make([]interface{}, total)
// 	for i := 0; i < total; i++ {
// 		// purposely skipping some data index
// 		if i == 8 || i == 13 || i == 114 || i == 211 || i == 350 || i == 418 || i == 476 {
// 			continue
// 		}
// 		amounts[i] = r.Float64() * 1000
// 	}

// 	amountsData := dataframe.NewSeriesFloat64("Amount", nil, amounts...)
// 	//fmt.Println(amountsData.Table())

// 	// combinig datetime and random amounts generated into a dataframe
// 	data := dataframe.NewDataFrame(dateTime, amountsData)

// 	hModel := HumanForecast(data)
// 	fmt.Println(hModel.data.Table())
// 	fmt.Println()

// 	alpha := 0.45
// 	beta := 0.03

// 	opts := HumanForecastOptions{Interval: Daily}
// 	hModelFit, err := hModel.Fit(ctx, alpha, beta, dataframe.Range{End: &[]int{335}[0]}, opts)
// 	if err != nil {
// 		t.Errorf("unexpected error: %s", err)
// 	}

// 	fmt.Println()
// 	spew.Dump(hModelFit)
// }
