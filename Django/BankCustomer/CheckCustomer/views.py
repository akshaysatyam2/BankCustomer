from django.shortcuts import render
import pickle

def home(request):
    return render(request, 'CheckCustomer/index.html')

def getPredictions(Country ,CreditScore,Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    ann = pickle.load(open('ml_model.sav', 'rb'))
    sc = pickle.load(open('scaler.sav', 'rb'))

    france = [1, 0, 0]
    spain= [0, 0, 1]
    germany = [0, 1, 0]


    if Country=='France':
        new_y_pred = ann.predict(sc.transform([[1, 0, 0 ,CreditScore,Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])) > 0.5
        print(new_y_pred)
        return new_y_pred

    elif Country == 'Spain':
        new_y_pred = ann.predict(sc.transform([[0, 0, 1,CreditScore,Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])) > 0.5
        print(new_y_pred)
        return new_y_pred

    elif Country == 'Germany':
        new_y_pred = ann.predict(sc.transform([[0, 1, 0,CreditScore,Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])) > 0.5 
        print(new_y_pred)
        return new_y_pred


def result(request):

    CreditScore = int(request.POST.get('CreditScore'))
    Country = (request.POST.get('Country'))
    Gender = int(request.POST.get('Gender'))
    Age = int(request.POST.get('Age'))
    Tenure = int(request.POST.get('Tenure'))
    Balance = int(request.POST.get('Balance'))
    NumOfProducts = int(request.POST.get('NumOfProducts'))
    HasCrCard = int(request.POST.get('HasCrCard'))
    IsActiveMember = int(request.POST.get('IsActiveMember'))
    EstimatedSalary = int(request.POST.get('EstimatedSalary'))


    print(f"Fetched {Country ,CreditScore,Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary}")

    result = getPredictions(Country ,CreditScore,Gender, Age, Tenure,Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
    print(result)
    return render(request, 'Checkcustomer/result.html', {'result': result})