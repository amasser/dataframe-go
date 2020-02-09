package forecast

import (
	"context"

	dataframe "github.com/rocketlaunchr/dataframe-go"
	pd "github.com/rocketlaunchr/dataframe-go/pandas"
)

// DataType is a const type used to
// specify data type selection from Sesmodel
type DataType int

const (
	// TrainData type specifies selecting trainData from sesModel
	TrainData DataType = 1
	// TestData type specifies selecting testData from sesModel
	TestData DataType = 2
	// MainData type specifies selecting the complete data from sesModel
	MainData DataType = 3
)

// Model is an interface to group trained models of Different
// Algorithms in the Forecast package under similar generic standard
type Model interface {
	// Fit Method performs the splitting and training of the Model Interface based on the Forecast algorithm Implemented.
	// It returns a trained Model ready to carry out future forecasts.
	// The argument α must be between [0,1]. Recent values receive more weight when α is closer to 1.
	Fit(context.Context, float64, ...dataframe.Range) (Model, error)

	// Predict method is used to run future predictions for Ses
	// Using Bootstrapping method
	Predict(context.Context, int) (*dataframe.SeriesFloat64, error)

	// Summary method is used to Print out Data Summary
	// From the Trained Model
	Summary()

	// Describe method
	Describe(context.Context, DataType, ...pd.DescribeOptions)
}
