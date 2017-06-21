

# Model Parameters

	Discount Factor
	Alpha
	Beta
	rSearch
	rWait

# Initialization
	
	initialValue['high'] = 0
	initialValue['low'] = 0

	terminateCheck = 10
	terminateCondition = 10^(-4)
	iteration = 0

#  In-place Value Iteration
	
	newValue = initialValue
	while (terminateCheck > terminateCondition)
		oldValue = newValue
		newValue['low'] 	= max over {	rSearch + dis*(alpha*oldValue['low'] + (1 - alpha)*oldValue['high']),
											rWait + dis*oldValue['low'] }

		newValue['high'] 	= max over {	beta*rSearch - 3*(1-beta) + dis*((1-beta)*oldValue[1] + beta*oldValue['high']),
									 		rWait + dis*oldValue['high'],
									 		dis*oldValue['low'] }

	 	terminateCheck 	= max over {	newValue['high'] - oldValue['high'],
	 										newValue['low'] - oldValue['low']	}
		iteration += 1
	while end

