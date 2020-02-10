package forecast

import (
	"context"

	dataframe "github.com/rocketlaunchr/dataframe-go"
	pd "github.com/rocketlaunchr/dataframe-go/pandas"
)

// DataType is a Enum used to
// specify data type selection from Sesmodel
type DataType int

const (
	// TrainData type specifies selection of trainData from sesModel
	TrainData DataType = 0
	// TestData type specifies selection of testData from sesModel
	TestData DataType = 1
	// MainData type specifies selection of the original data from sesModel
	MainData DataType = 2
)

// FitOptions is used to set necessary parameters
// needed to run Fit on a Model Algorithm
type FitOptions struct {
	Alpha          float64
	Beta           float64
	Gamma          float64
	Period         int
	ErrMtype       ErrorType
	TrainDataRange *dataframe.Range
}

// Model is an interface to group trained models of Different
// Algorithms in the Forecast package under similar generic standard
type Model interface {
	// Fit Method performs the splitting and training of the Model Interface based on the Forecast algorithm Implemented.
	// It returns a trained Model ready to carry out future forecasts.
	// The argument α must be between [0,1]. Recent values receive more weight when α is closer to 1.
	Fit(context.Context, *FitOptions) (Model, error)

	// Predict method is used to run future predictions for Ses
	// Using Bootstrapping method
	Predict(context.Context, int) (*dataframe.SeriesFloat64, error)

	// Summary method is used to Print out Data Summary
	// From the Trained Model
	Summary()

	// Describe method
	Describe(context.Context, DataType, ...pd.DescribeOptions)
}
