*** Settings ***
Documentation   Zoo robot testing
Library         Collections
Library         RequestsLibrary
Library         String
Library         OperatingSystem
Library         XML

*** Keywords ***
Operate Vertical
   [Documentation]               Post operation to configuring service. Operation allowed: deploy, stop, suspend, resume, clear, reset
   [Arguments]                   ${verticalId}       ${operation}                          ${expectStatus}
   Create Session                host                http://${ardaHost}:10021
   Log To Console                Operate vertical ${verticalId} with ${operation} ...
   ${resp}=                      Post Request        host                                  /vertical/${verticalId}/operation   data=${operation}
   ${statusCode}=                Convert To String   ${resp.status_code}
   Should Start With             ${statusCode}       20
   Wait Until Keyword Succeeds   10 min              5 sec                                 Status Equal                        ${verticalId}       ${expectStatus}

Status Equal
   [Documentation]                  Match certain vertical's status
   [Arguments]                      ${verticalId}                              ${status}
   Create Session                   host                                       http://${ardaHost}:10021
   Log To Console                   Get vertical ${verticalId}'s status ...
   ${resp}=                         Get Request                                host                        /vertical/${verticalId}
   ${statusCode}=                   Convert To String                          ${resp.status_code}
   Should Start With                ${statusCode}                              20
   ${json}=                         To Json                                    ${resp.content}
   Dictionary Should Contain Key    ${json}                                    status
   ${realStatus}=                   Get From Dictionary                        ${json}                     status
   Log To Console                   Expected=${status}, Actual=${realStatus}
   Should Be Equal As Strings       ${status}                                  ${realStatus}

Zoo Test
   [Arguments]         ${run_keyword}      ${verticals}
   @{verticalList}= 	 Split String 	     ${verticals}       separator=,
   :FOR                ${vertical}         IN                 @{verticalList}
   \                   Operate Vertical    ${vertical}        start              running
   \                   Run KeyWord         ${run_keyword}
   [Teardown]          Stop Verticals      @{verticalList}

Stop Verticals
   [Arguments]         @{verticalList}
   Remove Environment Variable             http_proxy
   :FOR                ${vertical}         IN                @{verticalList}
   \                   Operate Vertical    ${vertical}       stop               deployed/stopped

Prepare DataSource And Verticals
   Get Zoo Version
   Check Verticals

Check Verticals
   :FOR                   ${vertical}           IN             @{verticals}
   \                      Status Equal          ${vertical}    deployed/stopped

Run Shell
   [Arguments]       ${program}
   ${rc}             ${output}=     Run and Return RC and Output    ${program}
   Log To Console                   ${output}
   Should Be Equal As Integers      ${rc}          0

Get Zoo Version
   ${root}=               Parse XML           pom.xml
   ${version}=            Get Element Text    ${root}    version
   Log To Console         ${version}
   Set Global Variable    ${version}
   ${jar_path}=           Set Variable        ${jar_dir}/analytics-zoo-bigdl_${bigdl_version}-spark_${spark_version}-${version}-jar-with-dependencies.jar
   Set Global Variable    ${jar_path}