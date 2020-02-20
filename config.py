from collections import OrderedDict 

PROFESSIONAL_REMITTANCE_ADVICE = OrderedDict()

PROFESSIONAL_REMITTANCE_ADVICE['LASTNAME__PATIENT_ACCOUNT']             = ['LAST NAME','PATIENT ACCOUNT']
PROFESSIONAL_REMITTANCE_ADVICE['FIRSTNAME__MEMBER_ID']                  = ['FIRST NAME','MEMBER ID']
PROFESSIONAL_REMITTANCE_ADVICE['CLAIM_NUMBER__RECVDDT__SERVPROV']       = ['CLAIM NUMBER','RECVD DT','SERV PROV']
PROFESSIONAL_REMITTANCE_ADVICE['DATE_OF_SERVICE__FROM_THRU']            = ['FROM','THRU']
PROFESSIONAL_REMITTANCE_ADVICE['PROCEDURE__MODIFIER']                   = ['PROCEDURE','MODIFIER']
PROFESSIONAL_REMITTANCE_ADVICE['TOTAL_CHARGES']                         = ['TOTOL CHARGES']
PROFESSIONAL_REMITTANCE_ADVICE['PATIENT_NON_COVERED']                   = ['PATIENT NON COVERED']
PROFESSIONAL_REMITTANCE_ADVICE['NOTE']                                  = ['NOTE']
PROFESSIONAL_REMITTANCE_ADVICE['CONTRACT_WRITE_OFF']                    = ['CONTRACT WRITE OFF']
PROFESSIONAL_REMITTANCE_ADVICE['NOTE2']                                 = ['NOTE2']
PROFESSIONAL_REMITTANCE_ADVICE['PATIENT_DED_COPAY']                    = ['PATIENT DED/COPY']
PROFESSIONAL_REMITTANCE_ADVICE['PATIENT_COINS']                         = ['PATIENT COINS']
PROFESSIONAL_REMITTANCE_ADVICE['OTHER_INSURANCE_MEDICARE']             = ['OTHER INSURANCE MEDICARE']
PROFESSIONAL_REMITTANCE_ADVICE['CLIAM_PAID_INTEREST_PAID']             = ['CLAIM PAID/INTEREST PAID']
PROFESSIONAL_REMITTANCE_ADVICE['PATIENT_OWES']                          = ['PATIENT OWES']

PROFESSIONAL_REMITTANCE_ADVICE_CHECK = ['AMOUNT','NUMBER','DATE','ADDRESS']
PROFESSIONAL_REMITTANCE_ADVICE_META = [
    'LINE OF BUSINESS',
    'REMIT/CHECK DATE',
    'INTERNAL PROVIDER NUMBER',
    'NPI NUMBER',
    'TAX IDENTIFICATION NUMBER',
    'CHECK NUMBER',
    'REMITTANCE NUMBER',
    'PAGE NUMBER',
]


if __name__ == "__main__":
    print('--start--')

    print(PROFESSIONAL_REMITTANCE_ADVICE)
    print(PROFESSIONAL_REMITTANCE_ADVICE.keys())
    print(PROFESSIONAL_REMITTANCE_ADVICE.values())
    print(type(tuple(PROFESSIONAL_REMITTANCE_ADVICE.values()) ))
    print(tuple(PROFESSIONAL_REMITTANCE_ADVICE.values())[-1])