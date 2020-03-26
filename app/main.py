from ariadne import ObjectType, QueryType, MutationType, gql, make_executable_schema
from ariadne.asgi import GraphQL
from graphqlclient import GraphQLClient

# MDAO
import openmdao.api as om

# HTTP request library for access token call
import requests
# .env
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


def getAuthToken():
    authProvider = os.getenv('AUTH_PROVIDER')
    authDomain = os.getenv('AUTH_DOMAIN')
    authClientId = os.getenv('AUTH_CLIENT_ID')
    authSecret = os.getenv('AUTH_SECRET')
    authIdentifier = os.getenv('AUTH_IDENTIFIER')

    # Short-circuit for 'no-auth' scenario.
    if(authProvider == ''):
        print('Auth provider not set. Aborting token request...')
        return None

    url = ''
    if authProvider == 'keycloak':
        url = f'{authDomain}/auth/realms/{authIdentifier}/protocol/openid-connect/token'
    else:
        url = f'https://{authDomain}/oauth/token'

    payload = {
        'grant_type': 'client_credentials',
        'client_id': authClientId,
        'client_secret': authSecret,
        'audience': authIdentifier
    }

    headers = {'content-type': 'application/x-www-form-urlencoded'}

    r = requests.post(url, data=payload, headers=headers)
    response_data = r.json()
    print("Finished auth token request...")
    return response_data['access_token']


def getClient():

    graphqlClient = None

    # Build as closure to keep scope clean.

    def buildClient(client=graphqlClient):
        # Cached in regular use cases.
        if (client is None):
            print('Building graphql client...')
            token = getAuthToken()
            if (token is None):
                # Short-circuit for 'no-auth' scenario.
                print('Failed to get access token. Abandoning client setup...')
                return None
            url = os.getenv('MAANA_ENDPOINT_URL')
            client = GraphQLClient(url)
            client.inject_token('Bearer '+token)
        return client
    return buildClient()


# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
# Wrapping string in gql function provides validation and better error traceback
type_defs = gql("""
    type Query {
        solve(problem: Problem!): [Result!]!
    }

    input IndepVarComp {
        id: ID!
        value: Float!
    }

    input ExecComp {
        id: ID!
        type: String!
        eq: String!
    }

    input Driver {
        id: ID!
        optimizer: String!
    }

    input DesignVar {
        id: ID!
        lower: Float!
        upper: Float!
    }

    input Objective {
        id: ID!
    }

    input Problem {
        id: ID!
        driver: Driver!
        indeps: [IndepVarComp!]!
        exdep: ExecComp!
        designVars: [DesignVar!]!
        objective: Objective!
    }

    type Result {
        id: ID!
        value: Float!
    }
""")

# Map resolver functions to Query fields using QueryType
query = QueryType()

# Resolvers are simple python functions
@query.field("solve")
def resolve_solve(*_, problem):

    # # A resolver can access the graphql client via the context.
    # client = info.context["client"]

    # # Query all maana services.
    # result = client.execute('''
    # {
    #     allServices {
    #         id
    #         name
    #     }
    # }
    # ''')

    # print(result)

    # build the problem
    prob = om.Problem()

    # create the indeps
    indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
    for indep in problem['indeps']:
        indeps.add_output(indep['id'], indep['value'])

    # create the extdep
    prob.model.add_subsystem(
        problem['exdep']['type'], om.ExecComp(problem['exdep']['eq']))

    # connect the indeps to the extdep
    for indep in problem['indeps']:
        prob.model.connect(
            'indeps.' + indep['id'], problem['exdep']['type'] + '.' + indep['id'])

    # create the driver
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = problem['driver']['optimizer']

    # add the design variables
    for designVar in problem['designVars']:
        prob.model.add_design_var(
            designVar['id'], lower=designVar['lower'], upper=designVar['upper'])

    # add the objective
    prob.model.add_objective(problem['objective']['id'])

    prob.setup()
    prob.run_driver()

    results = [
        {'id': problem['objective']['id'],
            'value': prob[problem['objective']['id']]}
    ]

    for designVar in problem['designVars']:
        results.append({'id': designVar['id'], 'value': prob[designVar['id']]})

    return results


# Map resolver functions to custom type fields using ObjectType
# problem = ObjectType("Problem")

# @problem.field("fullName")
# def resolve_person_fullname(problem, *_):
#     return "%s %s" % (problem["firstName"], problem["lastName"])

# Create executable GraphQL schema
schema = make_executable_schema(type_defs, [query])

# --- ASGI app

# Create an ASGI app using the schema, running in debug mode
# Set context with authenticated graphql client.
app = GraphQL(
    schema, debug=True, context_value={'client': getClient()})
